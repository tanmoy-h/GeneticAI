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

    generated_text = processor.text_tokenizer.decode(output_ids[0], skip_special_tokens=True)

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
    n = len(results)
    correct = sum(r["is_correct"] for r in results)
    accuracy = correct / n if n else 0.0
    all_gt = [r["ground_truth"] for r in results]
    unique = list(set(all_gt))
    if len(unique) == 2:
        pos = unique[0]
        neg = unique[1]
        tp = sum(r["ground_truth"] == pos and r["is_correct"] for r in results)
        fp = sum(r["ground_truth"] == neg and r["predicted_answer"] == pos for r in results)
        fn = sum(r["ground_truth"] == pos and not r["is_correct"] for r in results)
        tn = sum(r["ground_truth"] == neg and r["is_correct"] for r in results)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return dict(accuracy=accuracy, precision=prec, recall=rec, f1_score=f1,
                    true_positives=tp, false_positives=fp,
                    true_negatives=tn, false_negatives=fn,
                    total_examples=n, correct_predictions=correct,
                    positive_label=pos, negative_label=neg)
    return dict(accuracy=accuracy, total_examples=n, correct_predictions=correct,
                unique_labels=unique)


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
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total:    {metrics['total_examples']}")
    print(f"Correct:  {metrics['correct_predictions']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    if "f1_score" in metrics:
        print(f"F1:       {metrics['f1_score']:.4f}")
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
    save_results(results, metrics, args.output_dir, tag=args.tag)


if __name__ == "__main__":
    main()
