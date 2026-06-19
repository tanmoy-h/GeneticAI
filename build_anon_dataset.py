"""
build_anon_dataset.py

Loads wanglab/kegg, anonymizes gene names, and saves a CSV with both
original and anonymized columns so you can verify before training.

The output CSV is used directly by train_dna_qwen.py via --kegg_anon_csv.

Usage:
    python build_anon_dataset.py
    python build_anon_dataset.py --output kegg_anon.csv --cache_dir ~/.cache/huggingface
"""
import re
import csv
import random
import argparse
from typing import Dict
from datasets import load_dataset

# ── Anonymization logic ───────────────────────────────────────────────────────

_NON_GENE_TERMS = {
    "ATP", "ADP", "AMP", "GTP", "GDP", "GMP", "DNA", "RNA", "PI3K",
    "IP3", "DAG", "ROS", "NO", "Ca2", "NAD", "NADH", "FADH",
}
_GENE_PATTERN = re.compile(r'\b([A-Z][A-Z0-9]{1,})\b')

# Matches molecule tokens between pathway operators // and ->
# e.g. "(GENE_1*) // cAMP -> (GENE_2)" → captures "cAMP"
_MOL_PATHWAY_PATTERN = re.compile(r'(?://|->)\s*([A-Za-z][A-Za-z0-9+\-]*)\s*(?:->|\(|$)')


def _build_gene_map(text: str) -> Dict[str, str]:
    """Scan text and return {gene_name: GENE_N} in order of first appearance."""
    gene_map: Dict[str, str] = {}
    counter = [1]

    def collect(match: re.Match) -> str:
        name = match.group(1)
        if name not in _NON_GENE_TERMS and name not in gene_map:
            gene_map[name] = f"GENE_{counter[0]}"
            counter[0] += 1
        return match.group(0)

    _GENE_PATTERN.sub(collect, text)
    return gene_map


def _build_mol_map(question: str) -> Dict[str, str]:
    """
    Extract molecule names from the pathway string in the question and
    return {molecule_name: MOL_N}. Molecules are tokens that appear between
    pathway operators (// and ->) and are not GENE_N tokens.
    """
    mol_map: Dict[str, str] = {}
    counter = [1]
    for match in _MOL_PATHWAY_PATTERN.finditer(question):
        name = match.group(1).strip()
        if name and not name.startswith("GENE_") and name not in mol_map:
            mol_map[name] = f"MOL_{counter[0]}"
            counter[0] += 1
    return mol_map


def _apply_mol_map(text: str, mol_map: Dict[str, str]) -> str:
    """Replace molecule names in text using whole-word matching."""
    for mol, token in sorted(mol_map.items(), key=lambda x: -len(x[0])):
        text = re.sub(r'\b' + re.escape(mol) + r'\b', token, text)
    return text


def _shuffle_gene_map(gene_map: Dict[str, str]) -> Dict[str, str]:
    """
    Randomly reassign GENE_N indices so the same gene gets a different
    index in each example, breaking positional patterns the LLM could learn.
    e.g. SNCA→GENE_1 in one example, SNCA→GENE_4 in another.
    """
    genes = list(gene_map.keys())
    indices = list(range(1, len(genes) + 1))
    random.shuffle(indices)
    return {gene: f"GENE_{idx}" for gene, idx in zip(genes, indices)}


def _apply_gene_map(text: str, gene_map: Dict[str, str]) -> str:
    def replace(match: re.Match) -> str:
        return gene_map.get(match.group(1), match.group(1))
    return _GENE_PATTERN.sub(replace, text)


# ─────────────────────────────────────────────────────────────────────────────

FIELDNAMES = [
    # ── identifiers ──────────────────────────────────────────────────────────
    "split",
    "answer",
    # ── original columns (for verification) ──────────────────────────────────
    "original_question",
    "original_reasoning",
    # ── anonymized columns (used for training) ────────────────────────────────
    "anon_question",
    "anon_reasoning",
    # ── maps (for verification) ───────────────────────────────────────────────
    "gene_map",
    "mol_map",
    # ── DNA sequences (unchanged) ─────────────────────────────────────────────
    "reference_sequence",
    "variant_sequence",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",    default="kegg_anon.csv")
    parser.add_argument("--cache_dir", default="~/.cache/huggingface")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Random seed for GENE_N index shuffling.")
    parser.add_argument("--no_shuffle", action="store_true",
                        help="Disable shuffling (use order-of-appearance instead).")
    parser.add_argument("--keep_chromosome", action="store_true",
                        help="Keep chromosome number in question (removed by default).")
    parser.add_argument("--anonymize_molecules", action="store_true",
                        help="Also replace pathway molecule names with MOL_N tokens "
                             "(e.g. cAMP→MOL_1). Use for Stage 0 curriculum training.")
    args = parser.parse_args()
    random.seed(args.seed)

    print("Loading wanglab/kegg ...")
    dataset = load_dataset("wanglab/kegg", cache_dir=args.cache_dir)

    rows = []
    for split_name, split_data in dataset.items():
        for ex in split_data:
            rows.append((split_name, ex))

    print(f"Writing {len(rows)} rows → {args.output}")
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for split_name, ex in rows:
            question  = ex["question"]
            reasoning = ex["reasoning"]

            # Remove chromosome number line unless explicitly kept
            if not args.keep_chromosome:
                question = re.sub(r'Chromosome\s+Number\s*:\s*\S+\s*', '', question,
                                  flags=re.IGNORECASE).strip()

            # Build gene map from question+reasoning combined, then shuffle.
            gene_map = _build_gene_map(question + " " + reasoning)
            if not args.no_shuffle:
                gene_map = _shuffle_gene_map(gene_map)
            anon_question  = _apply_gene_map(question,  gene_map)
            anon_reasoning = _apply_gene_map(reasoning, gene_map)

            # Optionally anonymize pathway molecules (Stage 0 curriculum).
            # Build mol_map from question AFTER gene replacement so GENE_N
            # tokens are not mistaken for molecules.
            if args.anonymize_molecules:
                mol_map = _build_mol_map(anon_question)
                anon_question  = _apply_mol_map(anon_question,  mol_map)
                anon_reasoning = _apply_mol_map(anon_reasoning, mol_map)
            else:
                mol_map = {}

            writer.writerow({
                "split":              split_name,
                "answer":             ex["answer"],
                "original_question":  question,
                "original_reasoning": reasoning,
                "anon_question":      anon_question,
                "anon_reasoning":     anon_reasoning,
                "gene_map":           str(gene_map),
                "mol_map":            str(mol_map),
                "reference_sequence": ex["reference_sequence"],
                "variant_sequence":   ex["variant_sequence"],
            })

    print(f"Done. Verify {args.output} then pass it to training with:")
    print(f"  --anonymize_genes {args.output}")


if __name__ == "__main__":
    main()
