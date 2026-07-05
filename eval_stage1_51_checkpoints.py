"""
eval_stage1_51_checkpoints.py — Parallel evaluation of Stage 1.51 gated checkpoints.

Identical in spirit to eval_stage1_5_checkpoints.py, but each checkpoint carries an
HRPO gate: model.pt + thinking_gate.pt + dna_injector.pt.  The gate is applied by
monkey-patching model.forward with _make_dual_mode_forward_w9 and pinning
_gate_warmup_factor = MAX_GATE_FACTOR — exactly as Stage 1.51 training does — so the
evaluated distribution matches what Stage 3 GRPO will start from.

Distributes checkpoints across GPUs (one subprocess per GPU) and ranks by answer
correctness.  Writes a results JSON whose "best" entry train_04_stage1_51.sh's
successor / Stage 3 can read to pick the peak-accuracy checkpoint instead of the
lowest-val-loss one.

Usage — 2 GPUs, 100 samples:
    python eval_stage1_51_checkpoints.py \\
        --stage1_ckpt  /scratch/.../stage1-sft-...ckpt \\
        --ckpt_dir     /scratch/.../train_04_stage1_51_anon \\
        --dna_cache    /scratch/.../dna_embeddings_kegg_2048.pt \\
        --gpus         0,1 --n_samples 100
"""

import os
import re
import sys
import glob
import json
import random
import argparse
import types
import datetime
from typing import List, Tuple, Optional

import torch
import torch.multiprocessing as mp
from transformers import StoppingCriteria, StoppingCriteriaList

from genomorph.models.dna_llm import DNALLMModel
from genomorph.models.evo2_tokenizer import register_evo2_tokenizer
from genomorph.dataset.utils import truncate_dna

# Gate modules — required for Stage 1.51 (unlike Stage 1.5 which has no gate).
from train_grpo_latent_reasoning import (
    ThinkingResidualGate,
    DNAHiddenInjector,
    _make_dual_mode_forward_w9,
    MAX_GATE_FACTOR,
)

register_evo2_tokenizer()

LATENT_START = "<start-latent>"
LATENT_END   = "<end-latent>"
LATENT_PAD   = "<latent>"

# Result tuple: (label, n_correct, n_total, accuracy, full_ckpt_path, details)
EvalResult = Tuple[str, int, int, float, str]


# ── Token helpers ─────────────────────────────────────────────────────────────

def ensure_latent_tokens(tokenizer, model) -> Tuple[int, int, int]:
    new_tokens = []
    for tok in (LATENT_START, LATENT_END, LATENT_PAD):
        if tokenizer.convert_tokens_to_ids(tok) == tokenizer.unk_token_id:
            new_tokens.append(tok)
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        model.text_model.resize_token_embeddings(len(tokenizer))
    start_id  = tokenizer.convert_tokens_to_ids(LATENT_START)
    end_id    = tokenizer.convert_tokens_to_ids(LATENT_END)
    latent_id = tokenizer.convert_tokens_to_ids(LATENT_PAD)
    return start_id, end_id, latent_id


# ── Data helpers ──────────────────────────────────────────────────────────────

def _trunc(ex: dict, per_side: int) -> tuple:
    if per_side > 0:
        ex = truncate_dna(ex, truncate_dna_per_side=per_side)
    return ex["reference_sequence"], ex["variant_sequence"]


def _make_row(question: str, reasoning: str, answer: str,
              ref_seq: str = "", var_seq: str = "") -> dict:
    user_text = (f"<|im_start|>user\n<|dna_pad|>\n<|dna_pad|>\n"
                 f"{question.strip()}\n<|im_end|>\n")
    asst_text = (f"<|im_start|>assistant\n<think>\n"
                 f"{reasoning.strip()}\n</think>\n")
    return {
        "text":          user_text + asst_text,
        "answer_raw":    answer,
        "dna_sequences": [ref_seq, var_seq],
    }


def load_val_rows(args) -> List[dict]:
    trunc = args.truncate_dna_per_side
    if args.kegg_csv:
        from genomorph.dataset.kegg import load_kegg_from_anon_csv
        ds = load_kegg_from_anon_csv(args.kegg_csv)
    else:
        from datasets import load_dataset
        ds = load_dataset(args.kegg_dataset, cache_dir=args.cache_dir)
    rows = [_make_row(ex["question"], ex["reasoning"], ex["answer"],
                      *_trunc(ex, trunc)) for ex in ds["val"]]
    print(f"[eval] Val rows: {len(rows)}")
    return rows


# ── Model init ────────────────────────────────────────────────────────────────

def _patch_dna_cache(model: DNALLMModel, cache_path: str, device: str):
    _c = torch.load(cache_path, map_location="cpu")

    def _cached_evo2_embed(self, input_ids: torch.Tensor, layer_name: str) -> torch.Tensor:
        key = input_ids.cpu().numpy().tobytes()
        emb = _c.get(key)
        if emb is None:
            raise KeyError(f"DNA not in cache (shape={input_ids.shape})")
        _p = next(self.dna_projection.parameters())
        return emb.to(device=_p.device, dtype=_p.dtype)

    model._evo2_embed = types.MethodType(_cached_evo2_embed, model)
    if model.dna_is_evo2 and getattr(model, "dna_model", None) is not None:
        model.dna_model.model.cpu()
        torch.cuda.empty_cache()


def init_base_model(args, device: str) -> DNALLMModel:
    """Build DNALLMModel, load Stage 1 weights, attach the HRPO gate, patch DNA cache.

    The gate (thinking_gate + dna_injector) is created once here and reused across
    checkpoints; evaluate_checkpoint reloads its weights per checkpoint. model.forward
    is patched with _make_dual_mode_forward_w9 and the gate factor pinned to
    MAX_GATE_FACTOR, matching Stage 1.51 training exactly.
    """
    print(f"[eval GPU:{device}] Initializing model ...", flush=True)
    model = DNALLMModel(
        text_model_name     = args.text_model_name,
        dna_model_name      = args.dna_model_name,
        cache_dir           = args.cache_dir,
        max_length_text     = args.max_length_text,
        max_length_dna      = args.max_length_dna,
        text_model_finetune = True,
        dna_model_finetune  = False,
        dna_is_evo2         = True,
        dna_embedding_layer = args.dna_embedding_layer,
        use_cross_attention = True,
        use_hrpo_gate       = False,   # gate is applied via the patched forward, not the built-in
        device              = device,
    ).to(device)
    model.text_model.config.use_cache = False

    ckpt  = torch.load(args.stage1_ckpt, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    clean = {(k[6:] if k.startswith("model.") else k): v for k, v in state.items()}
    missing, _ = model.load_state_dict(clean, strict=False)
    if missing:
        print(f"[eval GPU:{device}] Stage1 missing keys (expected for no gate): {missing[:2]}")

    ensure_latent_tokens(model.processor.tokenizer, model)

    # ── Attach gate + injector (weights reloaded per checkpoint) ──────────────
    hidden_size   = model.text_hidden_size
    thinking_gate = ThinkingResidualGate(
        hidden_size = hidden_size,
        use_ot_dist = True,   # must match Stage 1.51 / Stage 3 so ot_scale exists in ckpt
        r_min       = 0.5,
        r_max       = 0.99,
    ).to(device)
    dna_injector = DNAHiddenInjector(
        hidden_size = hidden_size,
        r_min       = 0.7,
        r_max       = 0.99,
    ).to(device)
    model.forward = types.MethodType(
        _make_dual_mode_forward_w9(thinking_gate, dna_injector),
        model,
    )
    model._gate_warmup_factor = MAX_GATE_FACTOR   # fixed, matches SFT (no ramp)
    model._eval_thinking_gate = thinking_gate     # stash for per-ckpt reload
    model._eval_dna_injector  = dna_injector
    print(f"[eval GPU:{device}] Gate attached (factor={MAX_GATE_FACTOR})", flush=True)

    if args.dna_cache:
        print(f"[eval GPU:{device}] Patching DNA cache: {args.dna_cache}", flush=True)
        _patch_dna_cache(model, args.dna_cache, device)
        cache_size = len(torch.load(args.dna_cache, map_location="cpu"))
        print(f"[eval GPU:{device}] Cache: {cache_size} sequences", flush=True)

    print(f"[eval GPU:{device}] Model ready.", flush=True)
    return model


def load_stage151_weights(model: DNALLMModel, ckpt_path: str, device: str):
    """Swap in Stage 1.51 weights (model.pt + thinking_gate.pt + dna_injector.pt)."""
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        n = len(missing)
        print(f"  [warn] {n} missing: {missing[:2]}{'…' if n > 2 else ''}", flush=True)
    if unexpected:
        n = len(unexpected)
        print(f"  [warn] {n} unexpected: {unexpected[:2]}{'…' if n > 2 else ''}", flush=True)

    ckpt_dir  = os.path.dirname(ckpt_path)
    gate_path = os.path.join(ckpt_dir, "thinking_gate.pt")
    inj_path  = os.path.join(ckpt_dir, "dna_injector.pt")
    if not (os.path.isfile(gate_path) and os.path.isfile(inj_path)):
        raise FileNotFoundError(
            f"Gate weights missing in {ckpt_dir} "
            f"(need thinking_gate.pt + dna_injector.pt)")
    model._eval_thinking_gate.load_state_dict(
        torch.load(gate_path, map_location=device))
    model._eval_dna_injector.load_state_dict(
        torch.load(inj_path, map_location=device))
    model._eval_thinking_gate.eval()
    model._eval_dna_injector.eval()


# ── Stopping criteria ─────────────────────────────────────────────────────────

class StopOnSecondThinkClose(StoppingCriteria):
    """Stop generation when </think> appears for the second time.
    First occurrence is legitimate (end of reasoning). Second means a loop."""
    def __init__(self, think_close_ids: List[int]):
        self.ids = think_close_ids
        self.n   = len(think_close_ids)

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        seq = input_ids[0].tolist()
        count = sum(
            1 for i in range(len(seq) - self.n + 1)
            if seq[i:i + self.n] == self.ids
        )
        return count >= 2


# ── Answer extraction ─────────────────────────────────────────────────────────

_EXPLANATION_DELIMS = (
    ' with ', ' due to', ' caused by', ' characterized by',
    ' resulting from', ' associated with', ' - ',
)

def extract_answer(generated: str) -> str:
    # Stop at newline / EOS so repetition loops don't pollute the prediction
    m = re.search(
        r'Answer:\s*(.+?)(?:<\|im_end\|>|<\|endoftext\|>|\n|\Z)', generated
    )
    if not m:
        return ""
    answer = m.group(1).strip().rstrip(".,;")
    # Trim verbose explanation appended after the disease name
    lower = answer.lower()
    for delim in _EXPLANATION_DELIMS:
        idx = lower.find(delim)
        if idx != -1:
            answer = answer[:idx].strip().rstrip(".,;")
            lower  = answer.lower()
    return answer


def is_correct(pred: str, gt: str) -> bool:
    if not pred:
        return False
    p, g = pred.lower().strip(), gt.lower().strip()
    return p in g or g in p


# ── Single-checkpoint evaluation ──────────────────────────────────────────────

def evaluate_checkpoint(
    model:          DNALLMModel,
    ckpt_path:      str,
    val_rows:       List[dict],
    sample_indices: List[int],
    device:         str,
    max_new_tokens: int = 400,
    gpu_tag:        str = "",
    print_samples:  int = 1,
) -> Tuple[int, int, List[Tuple[str, str, bool]]]:
    """
    Load ckpt_path (+ its gate) into model, run greedy generation on sample_indices,
    extract and score predicted answers.
    Returns (n_correct, n_total, [(pred, gt, correct), ...]).
    """
    load_stage151_weights(model, ckpt_path, device)
    model.eval()

    tokenizer = model.processor.tokenizer
    start_id, end_id, latent_id = ensure_latent_tokens(tokenizer, model)
    pad_id    = tokenizer.pad_token_id or 0
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_ids  = list({tid for tid in [tokenizer.eos_token_id, im_end_id]
                      if tid is not None and tid != tokenizer.unk_token_id})
    think_close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    stop_criteria   = StoppingCriteriaList([StopOnSecondThinkClose(think_close_ids)])

    think_tag = "<think>\n"
    n_correct = 0
    details: List[Tuple[str, str, bool]] = []
    prefix    = f"[{gpu_tag}] " if gpu_tag else ""

    for sample_num, idx in enumerate(sample_indices, 1):
        row       = val_rows[idx]
        gt_answer = row["answer_raw"]
        full_text = row["text"]
        dna_seqs  = row["dna_sequences"]

        cut         = full_text.find(think_tag)
        prompt_text = full_text[:cut + len(think_tag)] if cut != -1 else full_text

        batch = model.processor(
            text                = [prompt_text],
            batch_dna_sequences = [dna_seqs],
            return_tensors      = "pt",
            padding             = False,
            add_special_tokens  = False,
            max_length_text     = model.max_length_text,
            max_length_dna      = model.max_length_dna,
        )
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        dna_tok   = {k: v.to(device) for k, v in batch["dna_tokenized"].items()}
        idx_map   = list(batch["batch_idx_map"])

        try:
            with torch.no_grad():
                out_ids = model.generate(
                    input_ids            = input_ids,
                    attention_mask       = attn_mask,
                    dna_tokenized        = dna_tok,
                    batch_idx_map        = idx_map,
                    max_new_tokens       = max_new_tokens,
                    do_sample            = False,   # greedy — deterministic, repeatable
                    pad_token_id         = pad_id,
                    eos_token_id         = stop_ids,
                    bad_words_ids        = [[start_id], [end_id], [latent_id]],
                    stopping_criteria    = stop_criteria,
                )
        except KeyError as e:
            print(f"  {prefix}[skip] DNA cache miss sample={idx}: {e}", flush=True)
            details.append(("", gt_answer, False))
            continue

        # inputs_embeds generate returns only new tokens (no input prefix in out_ids)
        generated  = tokenizer.decode(out_ids[0], skip_special_tokens=False)
        # Strip everything from the second </think> onward (loop artifact)
        _tc = "</think>"
        _first = generated.find(_tc)
        if _first != -1:
            _second = generated.find(_tc, _first + len(_tc))
            if _second != -1:
                generated = generated[:_second]
        completion = "<think>\n" + generated
        pred       = extract_answer(generated)
        correct    = is_correct(pred, gt_answer)
        if correct:
            n_correct += 1
        details.append((pred, gt_answer, correct))

        mark = "✓" if correct else "✗"
        print(f"  {prefix}{mark} [{sample_num}/{len(sample_indices)}] "
              f"pred='{pred}'  gt='{gt_answer}'", flush=True)
        if sample_num <= print_samples:
            print(f"\n{prefix}── Sample {sample_num} full generation ──", flush=True)
            print(completion, flush=True)
            print(f"{prefix}────────────────────────────────", flush=True)
        elif not pred:
            tail = generated[-300:].replace("\n", "\\n")
            print(f"  {prefix}[debug] generated tail: ...{tail}", flush=True)

    return n_correct, len(sample_indices), details


# ── Worker subprocess ─────────────────────────────────────────────────────────

def worker_fn(gpu_id: int, ckpt_paths: List[str], val_rows: List[dict],
              sample_indices: List[int], args, result_queue):
    """
    Spawned subprocess: evaluates all assigned checkpoints on gpu_id sequentially.
    Puts List[EvalResult] into result_queue when done.
    """
    device  = f"cuda:{gpu_id}"
    gpu_tag = f"GPU{gpu_id}"
    print(f"[{gpu_tag}] Started — {len(ckpt_paths)} checkpoint(s) assigned.", flush=True)

    try:
        model = init_base_model(args, device)
    except Exception as exc:
        print(f"[{gpu_tag}] FATAL: model init failed: {exc}", flush=True)
        result_queue.put([])
        return

    worker_results: List[EvalResult] = []
    for ckpt_path in ckpt_paths:
        label = os.path.relpath(ckpt_path, args.ckpt_dir)
        print(f"[{gpu_tag}] ── {label} ──", flush=True)
        try:
            n_correct, n_total, details = evaluate_checkpoint(
                model          = model,
                ckpt_path      = ckpt_path,
                val_rows       = val_rows,
                sample_indices = sample_indices,
                device         = device,
                max_new_tokens = args.max_new_tokens,
                gpu_tag        = gpu_tag,
                print_samples  = args.print_samples,
            )
            acc = n_correct / n_total if n_total else 0.0
            print(f"[{gpu_tag}] {label}: {n_correct}/{n_total} ({acc*100:.1f}%)", flush=True)
            worker_results.append((label, n_correct, n_total, acc, ckpt_path, details))
        except Exception as exc:
            print(f"[{gpu_tag}] ERROR on {label}: {exc}", flush=True)
            worker_results.append((label, 0, len(sample_indices), 0.0, ckpt_path, []))

    result_queue.put(worker_results)
    print(f"[{gpu_tag}] Done.", flush=True)


# ── Checkpoint discovery ──────────────────────────────────────────────────────

def find_checkpoints(ckpt_dir: str, s_filter: Optional[List[int]]) -> List[str]:
    """
    Glob s??_pass??/model.pt under ckpt_dir (only dirs that also carry a gate).
    If s_filter is given (e.g. [4]), only include paths where sNN is in the set.
    """
    pattern = os.path.join(ckpt_dir, "s??_pass??", "model.pt")
    paths   = sorted(glob.glob(pattern))

    # Keep only checkpoints that carry gate weights (Stage 1.51 requirement)
    paths = [p for p in paths
             if os.path.isfile(os.path.join(os.path.dirname(p), "thinking_gate.pt"))]

    if s_filter:
        allowed = {f"s{n:02d}" for n in s_filter}
        paths = [p for p in paths
                 if os.path.basename(os.path.dirname(p))[:3] in allowed]

    return paths


def _distribute(items: list, n: int) -> List[list]:
    """Round-robin assignment of items to n workers."""
    chunks: List[list] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        chunks[i % n].append(item)
    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    _command   = " ".join(sys.argv)
    _timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Parse GPU list
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    if not gpu_ids:
        gpu_ids = [0]

    # Parse --s filter
    s_filter: Optional[List[int]] = None
    if args.s:
        s_filter = [int(x.strip()) for x in args.s.split(",") if x.strip()]

    # Discover checkpoints
    ckpt_paths = find_checkpoints(args.ckpt_dir, s_filter)
    if not ckpt_paths:
        hint = f" matching --s {args.s}" if args.s else ""
        raise FileNotFoundError(
            f"No s??_pass??/model.pt (with thinking_gate.pt){hint} in {args.ckpt_dir}")

    s_tag = f"  (s={args.s})" if args.s else ""
    print(f"[eval] Found {len(ckpt_paths)} checkpoint(s){s_tag}:")
    for p in ckpt_paths:
        print(f"  {os.path.relpath(p, args.ckpt_dir)}")

    # Load val rows in main process (pickled to workers)
    print("\n[eval] Loading val set ...")
    val_rows = load_val_rows(args)
    n = min(args.n_samples, len(val_rows))
    sample_indices = random.sample(range(len(val_rows)), n)
    print(f"[eval] Using {n} samples (seed={args.seed})")

    # Cap GPUs to number of checkpoints — no idle processes
    n_workers = min(len(gpu_ids), len(ckpt_paths))
    active_gpus = gpu_ids[:n_workers]
    chunks      = _distribute(ckpt_paths, n_workers)
    print(f"[eval] GPUs: {active_gpus}  |  checkpoints per GPU: {[len(c) for c in chunks]}")

    all_results: List[EvalResult] = []

    if n_workers == 1:
        # ── Single-process path (no spawn overhead) ───────────────────────────
        device = f"cuda:{active_gpus[0]}"
        model  = init_base_model(args, device)
        for ckpt_path in ckpt_paths:
            label = os.path.relpath(ckpt_path, args.ckpt_dir)
            print(f"\n[eval] ── {label} ──")
            n_correct, n_total, details = evaluate_checkpoint(
                model, ckpt_path, val_rows, sample_indices,
                device, args.max_new_tokens,
                print_samples=args.print_samples,
                gpu_tag=f"GPU{active_gpus[0]}",
            )
            acc = n_correct / n_total if n_total else 0.0
            print(f"  Accuracy: {n_correct}/{n_total}  ({acc*100:.1f}%)")
            all_results.append((label, n_correct, n_total, acc, ckpt_path, details))

    else:
        # ── Multi-process path: one subprocess per GPU ────────────────────────
        print(f"\n[eval] Spawning {n_workers} worker processes ...")
        ctx          = mp.get_context("spawn")
        result_queue = ctx.Queue()
        processes    = []

        for gpu_id, chunk in zip(active_gpus, chunks):
            p = ctx.Process(
                target = worker_fn,
                args   = (gpu_id, chunk, val_rows, sample_indices, args, result_queue),
                daemon = False,
            )
            p.start()
            processes.append(p)

        # Collect results — one put() per worker
        for _ in processes:
            batch = result_queue.get()
            all_results.extend(batch)

        for p in processes:
            p.join()

    # ── Ranked table ──────────────────────────────────────────────────────────
    all_results.sort(key=lambda x: x[3], reverse=True)

    def _f1_from_details(details):
        from sklearn.metrics import f1_score as sk_f1, precision_score, recall_score
        if not details:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        y_true = [gt for _, gt, _ in details]
        y_pred = [gt if ok else pred for pred, gt, ok in details]
        labels = sorted(set(y_true))
        prec_mac = float(precision_score(y_true, y_pred, labels=labels, average="macro",    zero_division=0))
        rec_mac  = float(recall_score(   y_true, y_pred, labels=labels, average="macro",    zero_division=0))
        f1_mac   = float(sk_f1(          y_true, y_pred, labels=labels, average="macro",    zero_division=0))
        prec_w   = float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
        rec_w    = float(recall_score(   y_true, y_pred, labels=labels, average="weighted", zero_division=0))
        f1_w     = float(sk_f1(          y_true, y_pred, labels=labels, average="weighted", zero_division=0))
        return prec_mac, rec_mac, f1_mac, prec_w, rec_w, f1_w

    sep = "=" * 70
    gpu_str = ",".join(str(g) for g in active_gpus)
    print(f"\n{sep}")
    print(f"  STAGE 1.51 CHECKPOINT RANKING{s_tag}  "
          f"(n={n}, seed={args.seed}, gpu={gpu_str})")
    print(sep)
    print(f"  {'Rank':<5} {'Accuracy':>10}  {'F1-mac':>8}  {'F1-wt':>8}  {'Correct':>9}  Checkpoint")
    print("  " + "-" * 72)
    for rank, (label, nc, nt, acc, _, details) in enumerate(all_results, 1):
        star = "  ★ BEST" if rank == 1 else ""
        _, _, f1_mac, _, _, f1_wt = _f1_from_details(details)
        print(f"  {rank:<5} {acc*100:>9.1f}%  {f1_mac:>8.4f}  {f1_wt:>8.4f}  {nc:>3}/{nt:<3}     {label}{star}")
    print(sep)

    if all_results:
        best_label, best_nc, best_nt, best_acc, best_path, best_details = all_results[0]
        prec_mac, rec_mac, f1_mac, prec_w, rec_w, f1_wt = _f1_from_details(best_details)
        print(f"\n  Best checkpoint : {best_path}")
        print(f"  Accuracy        : {best_nc}/{best_nt}  ({best_acc*100:.1f}%)")
        print(f"  Precision       : {prec_mac:.4f}  (macro)   {prec_w:.4f}  (weighted)")
        print(f"  Recall          : {rec_mac:.4f}  (macro)   {rec_w:.4f}  (weighted)")
        print(f"  F1              : {f1_mac:.4f}  (macro)   {f1_wt:.4f}  (weighted)\n")

    # ── Write results JSON ────────────────────────────────────────────────────
    if args.results_json:
        out = {
            "command":   _command,
            "timestamp": _timestamp,
            "args": {
                "stage1_ckpt":  args.stage1_ckpt,
                "ckpt_dir":     args.ckpt_dir,
                "s":            args.s,
                "gpus":         args.gpus,
                "n_samples":    n,
                "seed":         args.seed,
                "max_new_tokens": args.max_new_tokens,
                "kegg_dataset": args.kegg_dataset,
                "kegg_csv":     args.kegg_csv,
                "dna_cache":    args.dna_cache,
            },
            "ranked_results": [
                {
                    "rank":        rank,
                    "checkpoint":  label,
                    "full_path":   ckpt_path,
                    "n_correct":   nc,
                    "n_total":     nt,
                    "accuracy":    round(acc, 4),
                    "f1_macro":    round(_f1_from_details(det)[2], 4),
                    "f1_weighted": round(_f1_from_details(det)[5], 4),
                }
                for rank, (label, nc, nt, acc, ckpt_path, det) in enumerate(all_results, 1)
            ],
            "best": {
                "checkpoint":  all_results[0][0],
                "full_path":   all_results[0][4],
                "n_correct":   all_results[0][1],
                "n_total":     all_results[0][2],
                "accuracy":    round(all_results[0][3], 4),
                "f1_macro":    round(_f1_from_details(all_results[0][5])[2], 4),
                "f1_weighted": round(_f1_from_details(all_results[0][5])[5], 4),
            } if all_results else None,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.results_json)), exist_ok=True)
        with open(args.results_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[eval] Results written to: {args.results_json}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Parallel evaluation of Stage 1.51 gated LatentSp checkpoints by answer correctness"
    )
    p.add_argument("--stage1_ckpt",           required=True,
                   help="Stage 1 Lightning .ckpt to initialize model architecture")
    p.add_argument("--ckpt_dir",              required=True,
                   help="Root dir containing s??_pass??/{model.pt,thinking_gate.pt,dna_injector.pt}")
    p.add_argument("--s",                     default=None,
                   help="Curriculum steps to evaluate, e.g. '4'. Default: all s steps found.")
    p.add_argument("--gpus",                  default="0",
                   help="GPU IDs to use, e.g. '0,1,2,3'. One worker per GPU.")
    p.add_argument("--kegg_dataset",          default="wanglab/kegg")
    p.add_argument("--kegg_csv",              default=None,
                   help="Local anonymized CSV (overrides --kegg_dataset)")
    p.add_argument("--dna_cache",             default=None,
                   help="Precomputed DNA embeddings .pt (strongly recommended)")
    p.add_argument("--n_samples",             type=int, default=50,
                   help="Val examples per checkpoint (same samples for all)")
    p.add_argument("--max_new_tokens",        type=int, default=800)
    p.add_argument("--seed",                  type=int, default=42)
    p.add_argument("--text_model_name",       default="Qwen/Qwen3-1.7B")
    p.add_argument("--dna_model_name",        default="evo2_7b_base")
    p.add_argument("--dna_embedding_layer",   default="blocks.28.mlp.l3")
    p.add_argument("--cache_dir",             default="~/.cache/huggingface")
    p.add_argument("--max_length_text",       type=int, default=6000)
    p.add_argument("--max_length_dna",        type=int, default=2048)
    p.add_argument("--truncate_dna_per_side", type=int, default=1024)
    p.add_argument("--print_samples",         type=int, default=1,
                   help="Print full generation text for first N samples per checkpoint (default: 1)")
    p.add_argument("--results_json",          default=None,
                   help="Path to write ranked results as JSON (includes command + timestamp)")
    return p.parse_args()


if __name__ == "__main__":
    main()
