"""
Upload the anonymised KEGG CSV to HuggingFace as iitp-cse/kegg-anon-global.

Columns are renamed so the dataset is a drop-in replacement for wanglab/kegg:
  anon_question  -> question
  anon_reasoning -> reasoning
  answer, reference_sequence, variant_sequence are kept as-is.
The 'split' column is used to build train/val/test DatasetDict splits.

Usage (run from repo root after huggingface-cli login):
  python upload_anon_dataset.py
  python upload_anon_dataset.py --csv genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv
  python upload_anon_dataset.py --repo iitp-cse/kegg-anon-global --private
"""

import argparse
import csv

from datasets import Dataset, DatasetDict

KEEP_COLS = {"answer", "reference_sequence", "variant_sequence"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv")
    parser.add_argument("--repo", default="iitp-cse/kegg-anon-global")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    splits: dict = {}
    with open(args.csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            splits.setdefault(row["split"], []).append({
                "question":           row["anon_question"],
                "reasoning":          row["anon_reasoning"],
                "answer":             row["answer"],
                "reference_sequence": row["reference_sequence"],
                "variant_sequence":   row["variant_sequence"],
            })

    ds = DatasetDict({s: Dataset.from_list(rows) for s, rows in splits.items()})

    print(f"Splits: { {s: len(ds[s]) for s in ds} }")
    print(f"Columns: {ds[list(ds.keys())[0]].column_names}")
    print(f"Pushing to {args.repo} (private={args.private}) ...")

    ds.push_to_hub(args.repo, private=args.private)
    print("Done.")


if __name__ == "__main__":
    main()
