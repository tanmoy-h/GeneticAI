"""
Upload an anonymized KEGG CSV to the HuggingFace Hub as a dataset.

The uploaded dataset has the same schema as wanglab/kegg so it can be
passed directly to train_dna_qwen.py via --kegg_data_dir_huggingface
and to train_grpo_anon.py via --dataset_name.

Fields uploaded: question, reasoning, answer, reference_sequence, variant_sequence
Splits: whatever is in the CSV split column (train / val / test).

Usage:
    python upload_anon_dataset_to_hf.py \
        --csv_path /path/to/stage1_anon_genes_mol_keep_chr.csv \
        --repo_id  iitp-cse/kegg-anon-stage1 \
        [--token   hf_xxx]     # optional if already logged in via huggingface-cli
        [--private]            # make the dataset private
"""

import argparse
import csv
from datasets import Dataset, DatasetDict


def upload(csv_path: str, repo_id: str, token: str = None, private: bool = False):
    splits: dict = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split = row["split"]
            splits.setdefault(split, []).append({
                "question":           row["anon_question"],
                "reasoning":          row["anon_reasoning"],
                "answer":             row["answer"],
                "reference_sequence": row["reference_sequence"],
                "variant_sequence":   row["variant_sequence"],
            })

    dd = DatasetDict({name: Dataset.from_list(rows) for name, rows in splits.items()})
    print("Splits:", {k: len(v) for k, v in dd.items()})

    push_kwargs = {"private": private}
    if token:
        push_kwargs["token"] = token

    dd.push_to_hub(repo_id, **push_kwargs)
    print(f"Uploaded → https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload anonymized KEGG CSV to HuggingFace Hub.")
    parser.add_argument("--csv_path", required=True, help="Path to the anonymized KEGG CSV file.")
    parser.add_argument("--repo_id",  required=True, help="HF Hub repo to create, e.g. iitp-cse/kegg-anon-stage1")
    parser.add_argument("--token",    default=None,  help="HF write token (optional if already logged in).")
    parser.add_argument("--private",  action="store_true", help="Make the dataset private.")
    args = parser.parse_args()
    upload(args.csv_path, args.repo_id, args.token, args.private)
