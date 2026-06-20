"""
Upload the anonymised KEGG CSV to HuggingFace.

Columns are renamed so the dataset is a drop-in replacement for wanglab/kegg:
  anon_question  -> question
  anon_reasoning -> reasoning
  answer, reference_sequence, variant_sequence are kept as-is.
The 'split' column is used to build train/val/test DatasetDict splits.

Usage (run from repo root after huggingface-cli login):
  python upload_anon_dataset.py
  python upload_anon_dataset.py iitp-cse/kegg-anon-global
  python upload_anon_dataset.py iitp-cse/kegg-anon-global --private
  python upload_anon_dataset.py iitp-cse/kegg-anon-global --csv /path/to/other.csv
"""

import argparse
import csv

from datasets import Dataset, DatasetDict

DEFAULT_REPO = "iitp-cse/kegg-anon-global"
DEFAULT_CSV  = "genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", nargs="?", default=DEFAULT_REPO,
                        help=f"HuggingFace dataset repo (default: {DEFAULT_REPO})")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    print(f"CSV:     {args.csv}")
    print(f"Repo:    {args.repo}")
    print(f"Private: {args.private}")

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

    print(f"Splits:  { {s: len(ds[s]) for s in ds} }")
    print(f"Columns: {ds[list(ds.keys())[0]].column_names}")
    print(f"Pushing to {args.repo} ...")

    ds.push_to_hub(args.repo, private=args.private)
    print("Done.")


if __name__ == "__main__":
    main()
