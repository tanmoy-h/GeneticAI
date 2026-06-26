"""
test_latent_sft.py — Evaluate a single Stage 1.5 (LatentSp SFT) checkpoint.

Usage:
    python test_latent_sft.py \\
        --ckpt_path  /path/to/model.pt \\
        --split      val \\
        --kegg_dataset wanglab/kegg \\
        --output_dir /path/to/output \\
        --device     cuda
"""

import os
import re
import time
import argparse
from typing import List, Tuple

import torch
from sklearn.metrics import precision_score, recall_score, f1_score as sk_f1
from transformers import StoppingCriteria, StoppingCriteriaList

from genomorph.models.dna_llm import DNALLMModel
from genomorph.models.evo2_tokenizer import register_evo2_tokenizer
from genomorph.dataset.utils import truncate_dna

register_evo2_tokenizer()

LATENT_START = "<start-latent>"
LATENT_END   = "<end-latent>"
LATENT_PAD   = "<latent>"


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


class StopOnSecondThinkClose(StoppingCriteria):
    """Stop generation when </think> appears for the second time."""
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


_EXPLANATION_DELIMS = (
    ' with ', ' due to', ' caused by', ' characterized by',
    ' resulting from', ' associated with', ' - ',
)

def extract_answer(generated: str) -> str:
    m = re.search(
        r'Answer:\s*(.+?)(?:<\|im_end\|>|<\|endoftext\|>|\n|\Z)', generated
    )
    if not m:
        return ""
    answer = m.group(1).strip().rstrip(".,;")
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


def load_rows(args) -> List[dict]:
    trunc = getattr(args, "truncate_dna_per_side", 1024)
    if getattr(args, "kegg_csv", None):
        from genomorph.dataset.kegg import load_kegg_from_anon_csv
        ds = load_kegg_from_anon_csv(args.kegg_csv)
    else:
        from datasets import load_dataset
        ds = load_dataset(args.kegg_dataset, cache_dir=args.cache_dir)

    splits = ("val", "test") if args.split == "both" else (args.split,)
    rows: List[dict] = []
    for split in splits:
        for ex in ds[split]:
            if trunc > 0:
                ex = truncate_dna(ex, truncate_dna_per_side=trunc)
            ref_seq  = ex["reference_sequence"]
            var_seq  = ex["variant_sequence"]
            question = ex["question"]
            reasoning = ex["reasoning"]
            answer   = ex["answer"]
            user_text = (f"<|im_start|>user\n<|dna_pad|>\n<|dna_pad|>\n"
                         f"{question.strip()}\n<|im_end|>\n")
            asst_text = (f"<|im_start|>assistant\n<think>\n"
                         f"{reasoning.strip()}\n</think>\n")
            rows.append({
                "text":          user_text + asst_text,
                "answer_raw":    answer,
                "dna_sequences": [ref_seq, var_seq],
            })
    print(f"Loaded {len(rows)} examples (split={args.split})")
    return rows


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"Checkpoint:  {args.ckpt_path}")
    print(f"Split:       {args.split}")
    print(f"Device:      {device}")

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
        use_hrpo_gate       = False,
        device              = device,
    ).to(device)
    model.text_model.config.use_cache = False

    state = torch.load(args.ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    tokenizer = model.processor.tokenizer
    _emb = state.get("text_model.model.embed_tokens.weight")
    if _emb is not None and _emb.shape[0] != len(tokenizer):
        tokenizer.add_special_tokens({"additional_special_tokens":
                                      [LATENT_START, LATENT_END, LATENT_PAD]})
        model.text_model.resize_token_embeddings(_emb.shape[0])
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys: {missing[:3]}")
    print("Checkpoint loaded.")

    start_id, end_id, latent_id = ensure_latent_tokens(tokenizer, model)
    pad_id    = tokenizer.pad_token_id or 0
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_ids  = list({tid for tid in [tokenizer.eos_token_id, im_end_id]
                      if tid is not None and tid != tokenizer.unk_token_id})
    think_close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    stop_criteria   = StoppingCriteriaList([StopOnSecondThinkClose(think_close_ids)])

    rows = load_rows(args)
    model.eval()

    all_preds:   List[str]   = []
    all_targets: List[str]   = []
    gen_times:   List[float] = []
    correct = 0

    for i, row in enumerate(rows):
        gt_answer   = row["answer_raw"]
        full_text   = row["text"]
        dna_seqs    = row["dna_sequences"]
        think_tag   = "<think>\n"
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

        t0 = time.time()
        try:
            with torch.no_grad():
                out_ids = model.generate(
                    input_ids            = input_ids,
                    attention_mask       = attn_mask,
                    dna_tokenized        = dna_tok,
                    batch_idx_map        = idx_map,
                    max_new_tokens       = args.max_new_tokens,
                    do_sample            = False,
                    pad_token_id         = pad_id,
                    eos_token_id         = stop_ids,
                    bad_words_ids        = [[start_id], [end_id], [latent_id]],
                    stopping_criteria    = stop_criteria,
                )
        except KeyError as e:
            print(f"  [skip] DNA cache miss sample={i}: {e}")
            gen_times.append(time.time() - t0)
            all_preds.append("")
            all_targets.append(gt_answer)
            continue
        gen_times.append(time.time() - t0)

        generated = tokenizer.decode(out_ids[0], skip_special_tokens=False)
        _tc    = "</think>"
        _first = generated.find(_tc)
        if _first != -1:
            _second = generated.find(_tc, _first + len(_tc))
            if _second != -1:
                generated = generated[:_second]

        pred = extract_answer(generated)
        ok   = is_correct(pred, gt_answer)
        if ok:
            correct += 1
        all_preds.append(pred)
        all_targets.append(gt_answer)
        mark = "✓" if ok else "✗"
        print(f"  {mark} [{i+1}/{len(rows)}] pred='{pred}'  gt='{gt_answer}'")

    total_examples = len(all_targets)
    accuracy = correct / total_examples if total_examples else 0.0
    labels   = sorted(set(all_targets))
    prec_mac = precision_score(all_targets, all_preds, labels=labels, average="macro",    zero_division=0)
    rec_mac  = recall_score(   all_targets, all_preds, labels=labels, average="macro",    zero_division=0)
    f1_mac   = sk_f1(          all_targets, all_preds, labels=labels, average="macro",    zero_division=0)
    prec_w   = precision_score(all_targets, all_preds, labels=labels, average="weighted", zero_division=0)
    rec_w    = recall_score(   all_targets, all_preds, labels=labels, average="weighted", zero_division=0)
    f1_w     = sk_f1(          all_targets, all_preds, labels=labels, average="weighted", zero_division=0)
    avg_gen  = sum(gen_times) / len(gen_times) if gen_times else 0.0

    summary = (
        f"Test Results Summary:\n"
        f"Total examples:  {total_examples}\n"
        f"Correct:         {correct}\n"
        f"Classes:         {len(labels)}\n"
        f"Accuracy:        {accuracy:.4f}\n"
        f"Precision:       {prec_mac:.4f}  (macro)   {prec_w:.4f}  (weighted)\n"
        f"Recall:          {rec_mac:.4f}  (macro)   {rec_w:.4f}  (weighted)\n"
        f"F1:              {f1_mac:.4f}  (macro)   {f1_w:.4f}  (weighted)\n"
        f"Avg gen time:    {avg_gen:.2f}s/example\n"
        f"Total gen time:  {sum(gen_times)/60:.1f} min"
    )
    print(summary)

    os.makedirs(args.output_dir, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a Stage 1.5 LatentSp SFT checkpoint")
    p.add_argument("--ckpt_path",             required=True,
                   help="Path to model.pt checkpoint")
    p.add_argument("--output_dir",            required=True,
                   help="Directory to write results")
    p.add_argument("--split",                 default="val",
                   choices=["val", "test", "both"],
                   help="Dataset split to evaluate")
    p.add_argument("--kegg_dataset",          default="wanglab/kegg")
    p.add_argument("--kegg_csv",              default=None,
                   help="Local anonymized CSV (overrides --kegg_dataset)")
    p.add_argument("--text_model_name",       default="Qwen/Qwen3-1.7B")
    p.add_argument("--dna_model_name",        default="evo2_7b_base")
    p.add_argument("--dna_embedding_layer",   default="blocks.28.mlp.l3")
    p.add_argument("--max_length_text",       type=int, default=6000)
    p.add_argument("--max_length_dna",        type=int, default=2048)
    p.add_argument("--truncate_dna_per_side", type=int, default=1024)
    p.add_argument("--max_new_tokens",        type=int, default=800)
    p.add_argument("--cache_dir",             default="~/.cache/huggingface")
    p.add_argument("--device",                default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    main()
