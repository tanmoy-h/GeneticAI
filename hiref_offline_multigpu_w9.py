"""
Stage 2 (multi-GPU): Offline HiRef alignment over full KEGG dataset.
Week 9 version — supports both Stage 1 Lightning .ckpt AND Stage 1.5 plain .pt
state dicts from train_latent_sft.py.

Two modes:
  --shard_idx / --num_shards   Extract embeddings for one slice of the dataset.
  --merge                      Combine shard files and run HiRef.

Typical 2-GPU usage:
  python -u hiref_offline_multigpu_w9.py <common args> --num_shards 2 --shard_idx 0 --device cuda:0 &
  python -u hiref_offline_multigpu_w9.py <common args> --num_shards 2 --shard_idx 1 --device cuda:1 &
  wait
  python -u hiref_offline_multigpu_w9.py --output_dir stage2_output_w9 --num_shards 2 --merge

Or use the shell script:
  bash week9tests/sh_stage2_hiref_w9.sh
"""

import argparse
import contextlib
import io
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader

# ── BioReason imports ─────────────────────────────────────────────────────────
from genomorph.dataset.kegg import get_format_kegg_function, load_kegg_from_anon_csv, qwen_dna_collate_fn
from genomorph.dataset.utils import truncate_dna
from genomorph.models.dl.processing_dl import DLProcessor
from genomorph.models.dna_llm import DNALLMModel, get_target_modules
from genomorph.models.evo2_tokenizer import register_evo2_tokenizer
from train_dna_qwen import DNALLMFineTuner

from genomorph.hiref.HR_OT import HierarchicalRefinementOT
from genomorph.hiref.rank_annealing import optimal_rank_schedule

LATENT_TOKENS = ["<start-latent>", "<end-latent>", "<latent>"]


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model_for_hiref(ckpt_path: str, args) -> DNALLMModel:
    """Load model from .ckpt (Stage 1 Lightning) or .pt (Stage 1.5 plain state dict)."""
    register_evo2_tokenizer()
    if ckpt_path.endswith(".ckpt"):
        finetuner = DNALLMFineTuner.load_from_checkpoint(ckpt_path, map_location="cpu")
        return finetuner.model

    # Plain .pt from train_latent_sft.py
    print(f"  Detected plain .pt — constructing DNALLMModel from args...", flush=True)
    model = DNALLMModel(
        text_model_name=args.text_model_name,
        dna_model_name=args.dna_model_name,
        cache_dir=args.cache_dir,
        max_length_text=args.max_length_text,
        max_length_dna=args.max_length_dna,
        text_model_finetune=False,
        dna_model_finetune=False,
        dna_is_evo2=True,
        dna_embedding_layer=args.dna_embedding_layer,
        use_cross_attention=True,
        use_hrpo_gate=False,
        device="cpu",
    )

    tokenizer = model.text_tokenizer
    added = tokenizer.add_special_tokens({"additional_special_tokens": LATENT_TOKENS})
    if added:
        model.text_model.resize_token_embeddings(len(tokenizer))

    raw = torch.load(ckpt_path, map_location="cpu")
    state = raw.get("state_dict", raw)
    # Strip "model." prefix emitted by some Lightning wrappers
    state = {(k[6:] if k.startswith("model.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  WARNING: missing keys ({len(missing)}): {missing[:5]}", flush=True)
    if unexpected:
        print(f"  WARNING: unexpected keys ({len(unexpected)}): {unexpected[:5]}", flush=True)
    return model


# ── Embedding extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model: DNALLMModel, dataloader: DataLoader, device: str, batch_size: int = 1):
    model.eval()
    print(f"  Moving model to {device}...", flush=True)
    model.to(device)
    print(f"  Model on {device}. Starting forward passes...", flush=True)

    all_dna, all_ans, meta = [], [], []

    for batch_idx, batch in enumerate(dataloader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        dna_tokenized  = batch.get("dna_tokenized")
        batch_idx_map  = batch.get("batch_idx_map")

        if dna_tokenized is None:
            continue

        cur_bs = input_ids.shape[0]
        batch_dna_embeds = model.process_dna_embeddings(
            dna_tokenized, batch_idx_map, cur_bs
        )
        _p = next(model.dna_projection.parameters())

        u_dna_list = []
        for i in range(cur_bs):
            parts = []
            for slot in (i, i + cur_bs):
                if slot < len(batch_dna_embeds) and batch_dna_embeds[slot].shape[0] > 0:
                    parts.append(batch_dna_embeds[slot])
            if parts:
                u_dna_list.append(torch.cat(parts, dim=0).mean(dim=0))
            else:
                u_dna_list.append(torch.zeros(
                    model.text_hidden_size, device=_p.device, dtype=_p.dtype))

        u_dna = torch.stack(u_dna_list).to(device)

        answer_mask = labels != -100

        inputs_embeds, attn_mask = model.get_prompt_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dna_tokenized=dna_tokenized,
            batch_idx_map=batch_idx_map,
        )
        out = model.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = out.hidden_states[-1]

        def masked_mean(h, mask):
            mask_f = mask.unsqueeze(-1).float()
            return (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        answer_hidden = masked_mean(hidden, answer_mask)

        u_norm = F.normalize(u_dna.float(), dim=-1)
        a_norm = F.normalize(answer_hidden.float(), dim=-1)

        all_dna.append(u_norm.cpu())
        all_ans.append(a_norm.cpu())

        answers  = batch.get("answers",  [""] * cur_bs)
        kegg_ids = batch.get("kegg_ids", [""] * cur_bs)
        base_idx = len(meta)
        for i in range(cur_bs):
            meta.append({
                "row_idx": base_idx + i,
                "kegg_id": kegg_ids[i] if isinstance(kegg_ids, list) else str(kegg_ids[i]),
                "answer":  answers[i]  if isinstance(answers,  list) else str(answers[i]),
            })

        if (batch_idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

        if batch_idx == 0 or (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            seen = sum(t.shape[0] for t in all_dna)
            n_batches = len(dataloader)
            pct = 100.0 * (batch_idx + 1) / n_batches
            print(f"  [{batch_idx + 1}/{n_batches} batches | {seen} examples | {pct:.1f}%]", flush=True)

    dna_embs = torch.cat(all_dna, dim=0).numpy()
    ans_embs = torch.cat(all_ans, dim=0).numpy()
    return dna_embs, ans_embs, meta


# ── HiRef Monge map ───────────────────────────────────────────────────────────

def _linear_assignment_fallback(X: torch.Tensor, Y: torch.Tensor) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment
    print(f"  Computing {X.shape[0]}x{Y.shape[0]} cost matrix for linear assignment...", flush=True)
    C = torch.cdist(X, Y, p=2).numpy()
    row_ind, col_ind = linear_sum_assignment(C)
    monge_map = np.empty(X.shape[0], dtype=np.int64)
    monge_map[row_ind] = col_ind
    return monge_map


def run_hiref(dna_embs: np.ndarray, ans_embs: np.ndarray) -> np.ndarray:
    N = dna_embs.shape[0]
    X = torch.tensor(dna_embs, dtype=torch.float32)
    Y = torch.tensor(ans_embs, dtype=torch.float32)

    print(f"  Running HiRef on N={N} point pairs...", flush=True)

    clusters = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rank_schedule = optimal_rank_schedule(N)
        with contextlib.redirect_stdout(io.StringIO()):
            hiref = HierarchicalRefinementOT.init_from_point_clouds(
                X=X, Y=Y,
                rank_schedule=rank_schedule,
                sq_Euclidean=True,
                device="cpu",
            )
            clusters = hiref.run()
    except Exception as e:
        print(f"  HiRef failed ({e}), falling back to linear assignment (bijective).", flush=True)

    if clusters is not None:
        monge_map = np.full(N, -1, dtype=np.int64)
        for (idxX, idxY) in clusters:
            for i, j in zip(idxX.numpy(), idxY.numpy()):
                monge_map[int(i)] = int(j)

        unmapped = int((monge_map == -1).sum())
        if unmapped > 0:
            print(f"  Warning: {unmapped} HiRef-unmapped points — filling with linear assignment.", flush=True)
            assigned = set(monge_map[monge_map >= 0].tolist())
            free_targets = [j for j in range(N) if j not in assigned]
            unmapped_idx = np.where(monge_map == -1)[0]
            X_un = X[unmapped_idx]
            Y_free = Y[free_targets]
            C_sub = torch.cdist(X_un, Y_free, p=2).numpy()
            from scipy.optimize import linear_sum_assignment
            r, c = linear_sum_assignment(C_sub)
            for ri, ci in zip(r, c):
                monge_map[unmapped_idx[ri]] = free_targets[ci]
    else:
        monge_map = _linear_assignment_fallback(X, Y)

    unique = len(set(monge_map.tolist()))
    print(f"  Monge map done. Unique targets: {unique}/{N} ({100*unique/N:.1f}% bijective)", flush=True)
    return monge_map


# ── Merge shards and run HiRef ────────────────────────────────────────────────

def merge_and_run_hiref(args):
    print(f"Merging {args.num_shards} shards from {args.output_dir}...", flush=True)
    all_dna, all_ans, all_meta = [], [], []

    for s in range(args.num_shards):
        dna  = np.load(os.path.join(args.output_dir, f"dna_embeddings_shard{s}.npy"))
        ans  = np.load(os.path.join(args.output_dir, f"answer_embeddings_shard{s}.npy"))
        with open(os.path.join(args.output_dir, f"metadata_shard{s}.json")) as f:
            meta = json.load(f)
        all_dna.append(dna)
        all_ans.append(ans)
        all_meta.extend(meta)
        print(f"  shard {s}: {dna.shape[0]} examples", flush=True)

    dna_embs = np.concatenate(all_dna, axis=0)
    ans_embs = np.concatenate(all_ans, axis=0)
    N = dna_embs.shape[0]
    print(f"Merged: dna_embs {dna_embs.shape}, ans_embs {ans_embs.shape}", flush=True)

    monge_map = run_hiref(dna_embs, ans_embs)
    print(f"Monge map computed. Sample: {monge_map[:10]}")

    for i, m in enumerate(all_meta):
        m["monge_target"] = int(monge_map[i])

    np.save(os.path.join(args.output_dir, "dna_embeddings.npy"), dna_embs)
    np.save(os.path.join(args.output_dir, "answer_embeddings.npy"), ans_embs)
    np.save(os.path.join(args.output_dir, "monge_map.npy"), monge_map)
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(all_meta, f, indent=2)

    print(f"\nStage 2 complete. Outputs saved to: {args.output_dir}")
    print(f"  dna_embeddings.npy      {dna_embs.shape}")
    print(f"  answer_embeddings.npy   {ans_embs.shape}")
    print(f"  monge_map.npy           {monge_map.shape}  (unique targets: {len(set(monge_map.tolist()))})")
    print(f"  metadata.json           {N} entries")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    if args.merge:
        merge_and_run_hiref(args)
        return

    print(f"[shard {args.shard_idx}] Loading checkpoint: {args.ckpt_path}", flush=True)
    model = load_model_for_hiref(args.ckpt_path, args)
    model.eval()

    if args.kegg_dataset:
        from datasets import load_dataset as _load_hf
        print(f"[shard {args.shard_idx}] Loading HF dataset: {args.kegg_dataset}", flush=True)
        dataset = _load_hf(args.kegg_dataset, cache_dir=args.cache_dir)
    else:
        print(f"[shard {args.shard_idx}] Loading CSV dataset: {args.kegg_csv}", flush=True)
        dataset = load_kegg_from_anon_csv(args.kegg_csv)
    dataset = dataset.map(get_format_kegg_function("dna-llm"))

    from datasets import concatenate_datasets
    train_splits = [v for k, v in dataset.items() if "train" in k.lower()]
    full_dataset = concatenate_datasets(train_splits) if train_splits else concatenate_datasets(list(dataset.values()))
    print(f"[shard {args.shard_idx}] Using splits: {[k for k in dataset.keys() if 'train' in k.lower()] or list(dataset.keys())} ({len(full_dataset)} examples)", flush=True)

    if args.truncate_dna_per_side:
        full_dataset = full_dataset.map(
            truncate_dna,
            fn_kwargs={"truncate_dna_per_side": args.truncate_dna_per_side}
        )

    shard_size = (len(full_dataset) + args.num_shards - 1) // args.num_shards
    start = args.shard_idx * shard_size
    end   = min(start + shard_size, len(full_dataset))
    shard_dataset = full_dataset.select(range(start, end))
    print(f"[shard {args.shard_idx}] examples {start}–{end-1} ({len(shard_dataset)} examples)", flush=True)

    processor = DLProcessor(
        tokenizer=model.text_tokenizer,
        dna_tokenizer=model.dna_tokenizer,
    )
    collate_fn = partial(
        qwen_dna_collate_fn,
        processor=processor,
        max_length_text=args.max_length_text,
        max_length_dna=args.max_length_dna,
        return_answer_in_batch=True,
        truncate_for_generation=False,
    )
    dataloader = DataLoader(
        shard_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    print(f"[shard {args.shard_idx}] Extracting embeddings...", flush=True)
    dna_embs, ans_embs, meta = extract_embeddings(model, dataloader, args.device, args.batch_size)
    print(f"[shard {args.shard_idx}] Extracted: {dna_embs.shape}", flush=True)

    for i, m in enumerate(meta):
        m["row_idx"] = start + i

    np.save(os.path.join(args.output_dir, f"dna_embeddings_shard{args.shard_idx}.npy"), dna_embs)
    np.save(os.path.join(args.output_dir, f"answer_embeddings_shard{args.shard_idx}.npy"), ans_embs)
    with open(os.path.join(args.output_dir, f"metadata_shard{args.shard_idx}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[shard {args.shard_idx}] Saved. Run --merge after all shards complete.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 (multi-GPU): Offline HiRef alignment — w9")
    # Checkpoint / data
    parser.add_argument("--ckpt_path",              default=None,
                        help="Path to checkpoint (.ckpt Lightning or .pt plain state dict)")
    parser.add_argument("--kegg_csv",               default=None,
                        help="Path to local anonymized CSV (alternative to --kegg_dataset)")
    parser.add_argument("--kegg_dataset",           default=None,
                        help="HuggingFace dataset name, e.g. wanglab/kegg")
    parser.add_argument("--output_dir",             default="stage2_output_w9")
    # Shard control
    parser.add_argument("--device",                 default="cuda")
    parser.add_argument("--batch_size",             type=int, default=2)
    parser.add_argument("--num_shards",             type=int, default=2)
    parser.add_argument("--shard_idx",              type=int, default=0)
    parser.add_argument("--merge",                  action="store_true")
    # Model config — required for plain .pt, ignored for .ckpt
    parser.add_argument("--text_model_name",        default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dna_model_name",         default="evo2_7b_base")
    parser.add_argument("--dna_embedding_layer",    default="blocks.28.mlp.l3")
    parser.add_argument("--max_length_text",        type=int, default=6000)
    parser.add_argument("--max_length_dna",         type=int, default=2048)
    parser.add_argument("--truncate_dna_per_side",  type=int, default=1024)
    parser.add_argument("--cache_dir",              default="~/.cache/huggingface")
    args = parser.parse_args()

    if not args.merge and args.ckpt_path is None:
        parser.error("--ckpt_path is required unless --merge is set")
    if not args.merge and args.kegg_csv is None and args.kegg_dataset is None:
        parser.error("--kegg_csv or --kegg_dataset is required unless --merge is set")

    main(args)
