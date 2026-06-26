#!/usr/bin/env python
"""
Evaluation script for GRPO-trained BioReason model on KEGG test set.
Loads a GRPO checkpoint (pytorch_model.bin) and evaluates on val+test.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any, Optional
import time
import pandas as pd
from datetime import datetime
from pathlib import Path

import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset, concatenate_datasets
from transformers import GenerationConfig

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bioreason.models.dna_llm import DNALLMModel, get_target_modules
from bioreason.models.dl.processing_dl import DLProcessor
from bioreason.models.evo2_tokenizer import Evo2Tokenizer, register_evo2_tokenizer
from bioreason.dataset.utils import truncate_dna
from bioreason.dataset.kegg import format_kegg_for_dna_llm
from trl.data_utils import maybe_apply_chat_template

register_evo2_tokenizer()


# ── Model initialisation ────────────────────────────────────────────────────────

def _setup_lora(model: DNALLMModel, lora_r: int, lora_alpha: int, lora_dropout: float):
    target_modules = get_target_modules(model)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        init_lora_weights="gaussian",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.text_model = prepare_model_for_kbit_training(model.text_model)
    model.text_model = get_peft_model(model.text_model, lora_config)
    return model


def load_model(args) -> DNALLMModel:
    print("Initialising DNALLMModel …")
    model = DNALLMModel(
        text_model_name=args.text_model_name,
        dna_model_name=args.dna_model_name,
        cache_dir=args.cache_dir,
        max_length_dna=args.max_length_dna,
        max_length_text=args.max_length_text,
        text_model_finetune=True,
        dna_model_finetune=False,
        dna_is_evo2=args.dna_is_evo2,
        dna_embedding_layer=args.dna_embedding_layer,
        device="cuda",
    )

    model = _setup_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)

    ckpt_file = Path(args.grpo_checkpoint) / "pytorch_model.bin"
    print(f"Loading GRPO checkpoint: {ckpt_file}")
    state_dict = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    print(f"Checkpoint has {len(state_dict)} keys")

    # Resize embeddings if vocab sizes differ
    for k, v in state_dict.items():
        if "embed_tokens" in k and "weight" in k:
            ckpt_vocab = v.shape[0]
            cur_vocab = len(model.text_tokenizer)
            if ckpt_vocab != cur_vocab:
                print(f"Resizing embeddings {cur_vocab} → {ckpt_vocab}")
                if hasattr(model.text_model, "base_model"):
                    model.text_model.base_model.model.resize_token_embeddings(ckpt_vocab)
                else:
                    model.text_model.resize_token_embeddings(ckpt_vocab)
            break

    result = model.load_state_dict(state_dict, strict=False)
    print(f"load_state_dict → missing {len(result.missing_keys)} | unexpected {len(result.unexpected_keys)}")

    model = model.cuda()
    model.eval()
    print("✅ Model ready on GPU")
    return model


# ── Data loading ────────────────────────────────────────────────────────────────

def load_test_examples(dataset_name: str, truncate_dna_per_side: int) -> List[Dict]:
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name)
    splits = []
    for split in ("val", "test"):
        if split in ds:
            splits.append(ds[split])
    combined = concatenate_datasets(splits)
    print(f"Loaded {len(combined)} examples (val+test)")

    if truncate_dna_per_side > 0:
        combined = combined.map(
            truncate_dna, fn_kwargs={"truncate_dna_per_side": truncate_dna_per_side}
        )

    examples = [format_kegg_for_dna_llm(ex, is_sft=False) for ex in combined]
    return examples


# ── Inference ───────────────────────────────────────────────────────────────────

def run_one(model: DNALLMModel, processor: DLProcessor,
            example: Dict, gen_config: GenerationConfig) -> Dict:
    prompts_text = [maybe_apply_chat_template(example, processor)["prompt"]]
    prepared = processor(
        text=prompts_text,
        batch_dna_sequences=[example["dna_sequences"]],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
        max_length_text=model.max_length_text,
        max_length_dna=model.max_length_dna,
    )
    prepared = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                for k, v in prepared.items()}

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=prepared["input_ids"],
            attention_mask=prepared["attention_mask"],
            dna_tokenized=prepared.get("dna_tokenized"),
            batch_idx_map=prepared.get("batch_idx_map"),
            generation_config=gen_config,
        )

    generated_text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract answer: look for text after </think> then after "Answer:"
    extracted = generated_text
    if "</think>" in extracted:
        extracted = extracted.split("</think>")[-1]
    predicted = ""
    if "Answer:" in extracted:
        predicted = extracted.split("Answer:")[-1].strip()
    elif extracted.strip():
        predicted = extracted.strip()

    predicted = predicted.lower()
    ground_truth = example["answer"].strip().lower()
    for ch in ".,!?\"'":
        predicted = predicted.replace(ch, "")
        ground_truth = ground_truth.replace(ch, "")

    is_correct = ground_truth in predicted
    return {
        "generated_text": generated_text,
        "predicted_answer": predicted,
        "ground_truth": ground_truth,
        "is_correct": is_correct,
        "question": str(example.get("prompt", "")),
    }


# ── Metrics ─────────────────────────────────────────────────────────────────────

def calculate_metrics(results: List[Dict]) -> Dict:
    from sklearn.metrics import precision_score, recall_score, f1_score
    n = len(results)
    correct = sum(r["is_correct"] for r in results)
    accuracy = correct / n if n else 0.0

    y_true = [r["ground_truth"] for r in results]
    # For substring-match eval, map predicted to gt label when it contains it
    y_pred = [r["ground_truth"] if r["is_correct"] else r["predicted_answer"] for r in results]

    labels = sorted(set(y_true))
    prec_mac = precision_score(y_true, y_pred, labels=labels, average="macro",    zero_division=0)
    rec_mac  = recall_score(   y_true, y_pred, labels=labels, average="macro",    zero_division=0)
    f1_mac   = f1_score(       y_true, y_pred, labels=labels, average="macro",    zero_division=0)
    prec_w   = precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    rec_w    = recall_score(   y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    f1_w     = f1_score(       y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    return dict(
        accuracy=accuracy,
        macro_precision=float(prec_mac),
        macro_recall=float(rec_mac),
        f1_macro=float(f1_mac),
        precision_weighted=float(prec_w),
        recall_weighted=float(rec_w),
        f1_weighted=float(f1_w),
        total_examples=n,
        correct_predictions=correct,
        num_classes=len(labels),
    )


def save_results(results: List[Dict], metrics: Dict, output_dir: str, tag: str = "grpo"):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(output_dir, f"{tag}_{ts}")

    with open(f"{base}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{base}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    df = pd.DataFrame([{
        "question": r["question"],
        "predicted_answer": r["predicted_answer"],
        "ground_truth": r["ground_truth"],
        "is_correct": r["is_correct"],
        "generated_text": r["generated_text"],
    } for r in results])
    df.to_csv(f"{base}_results.csv", index=False)

    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)
    print(f"Total examples:  {metrics['total_examples']}")
    print(f"Correct:         {metrics['correct_predictions']}")
    print(f"Classes:         {metrics['num_classes']}")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Precision:       {metrics['macro_precision']:.4f}  (macro)   {metrics['precision_weighted']:.4f}  (weighted)")
    print(f"Recall:          {metrics['macro_recall']:.4f}  (macro)   {metrics['recall_weighted']:.4f}  (weighted)")
    print(f"F1:              {metrics['f1_macro']:.4f}  (macro)   {metrics['f1_weighted']:.4f}  (weighted)")
    print(f"Avg gen time:    {metrics.get('avg_gen_time_s', 0):.2f}s/example")
    print(f"Total gen time:  {metrics.get('total_time_s', 0)/60:.1f} min")
    print("=" * 60)
    print(f"Results: {base}_results.csv")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate GRPO BioReason on KEGG test set")
    parser.add_argument("--grpo_checkpoint",   required=True)
    parser.add_argument("--dataset_name",      default="wanglab/kegg")
    parser.add_argument("--text_model_name",   default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dna_model_name",    default="evo2_7b_base")
    parser.add_argument("--dna_is_evo2",       type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--dna_embedding_layer", default="blocks.28.mlp.l3")
    parser.add_argument("--cache_dir",         default=None)
    parser.add_argument("--max_length_dna",    type=int, default=2048)
    parser.add_argument("--max_length_text",   type=int, default=1024)
    parser.add_argument("--truncate_dna_per_side", type=int, default=1024)
    parser.add_argument("--lora_r",            type=int, default=32)
    parser.add_argument("--lora_alpha",        type=int, default=64)
    parser.add_argument("--lora_dropout",      type=float, default=0.0)
    parser.add_argument("--max_new_tokens",    type=int, default=800)
    parser.add_argument("--temperature",       type=float, default=0.0)
    parser.add_argument("--top_p",             type=float, default=0.95)
    parser.add_argument("--top_k",             type=int, default=20)
    parser.add_argument("--output_dir",        default="./eval_results")
    parser.add_argument("--max_examples",      type=int, default=None)
    parser.add_argument("--tag",               default="grpo")
    args = parser.parse_args()

    print("=" * 60)
    print("BioReason GRPO Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.grpo_checkpoint}")
    print(f"Dataset:    {args.dataset_name}")
    print(f"Output:     {args.output_dir}")

    examples = load_test_examples(args.dataset_name, args.truncate_dna_per_side)
    if args.max_examples:
        examples = examples[:args.max_examples]
        print(f"Capped at {len(examples)} examples")

    model = load_model(args)
    processor = DLProcessor(
        tokenizer=model.text_tokenizer,
        dna_tokenizer=model.dna_tokenizer,
    )

    do_sample = args.temperature > 0
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        temperature=args.temperature if do_sample else None,
        top_p=args.top_p if do_sample else None,
        top_k=args.top_k if do_sample else None,
        pad_token_id=model.text_tokenizer.pad_token_id,
        eos_token_id=model.text_tokenizer.eos_token_id,
    )

    results = []
    total = len(examples)
    gen_times = []
    for ex in examples:
        t0 = time.time()
        try:
            res = run_one(model, processor, ex, gen_config)
        except Exception as e:
            print(f"Error on example: {e}")
            res = {"generated_text": "", "predicted_answer": "",
                   "ground_truth": ex["answer"].strip().lower(),
                   "is_correct": False, "question": ""}
        elapsed = time.time() - t0
        gen_times.append(elapsed)
        results.append(res)

        n = len(results)
        correct = sum(r["is_correct"] for r in results)
        mark = "✓" if res["is_correct"] else "✗"
        gt_field = f"gt='{res['ground_truth']}'"
        pred_field = f"pred='{res['predicted_answer']}'"
        print(f"[TEST] {n:4d} | {mark} | {gt_field:<45} | {pred_field} | {elapsed:.2f}s")
        avg_gen = sum(gen_times) / len(gen_times)
        print(f"[TEST] Batch {n}/{total} done — acc so far: {correct}/{n} ({100*correct/n:.1f}%) | avg_gen: {avg_gen:.2f}s")

    metrics = calculate_metrics(results)
    metrics["avg_gen_time_s"] = sum(gen_times) / len(gen_times) if gen_times else 0.0
    metrics["total_time_s"] = sum(gen_times)
    save_results(results, metrics, args.output_dir, tag=args.tag)


if __name__ == "__main__":
    main()
