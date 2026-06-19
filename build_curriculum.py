"""
build_curriculum.py

Generates CSVs for all curriculum training stages and test configurations.

Curriculum design is driven by empirical results:
  - Linear projection is the only architecture that helps with anonymous text (+5%)
  - Cross-attention and CLIP both hurt with anonymous text (-8% to -13%)
  - Cross-attention helps with real gene names (+4-7%)
  - Shuffle vs no-shuffle: negligible difference (0.1% F1)
  - Global anon helps LLM-only (+11%) but not DNA+LLM (+0.8%)

Curriculum stages (anonymous -> real -> DNA only)
─────────────────────────────────────────────────
  Stage 0  Anon genes + mol, no Chr       Architecture: Linear
  Stage 1  Anon genes + mol + Chr         Architecture: Linear
  Stage 2  Real genes + pathway + Chr     Architecture: migrate Linear -> Cross-attn
  Stage 3  Real genes + pathway, no Chr   Architecture: Cross-attn
  Stage 4  Anon question + DNA only       Architecture: migrate Cross-attn -> Linear

Architecture migration
──────────────────────
  Stage 1 -> 2: python migrate_checkpoint.py --src stage1/best.ckpt --dst stage2_init.ckpt
  Stage 3 -> 4: python migrate_checkpoint.py --src stage3/best.ckpt --dst stage4_init.ckpt
                (reverse migration: cross-attn -> linear, reinit dna_projection)

Test configurations
───────────────────
  Test 1   Real, Chr, pathway             HF dataset — no CSV
  Test 2   Anon, Chr, pathway             global_test2_anon_keep_chr.csv
  Test 3   Anon, no Chr, pathway          global_test3_anon_no_chr.csv
  Test 4   Anon, no Chr, no pathway       global_test4_anon_no_chr_no_pathway.csv
  Test 5   Anon question + DNA only       global_test5_anon_question_dna_only.csv

Usage:
    python build_curriculum.py                     # global maps (default) — all stages + tests
    python build_curriculum.py --shuffle           # per-example shuffled maps
    python build_curriculum.py --stages 0 1        # selected stages only
    python build_curriculum.py --tests  2 3 4      # selected tests only
    python build_curriculum.py --prefix my_run     # custom filename prefix
    python build_curriculum.py --cache_dir ~/.cache/huggingface
    python build_curriculum.py --seed 123
"""

import re
import csv
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset

# ── Regex patterns ────────────────────────────────────────────────────────────

# Metabolite / non-gene abbreviations — excluded from gene map
_NON_GENE_TERMS = {
    "ATP", "ADP", "AMP", "GTP", "GDP", "GMP", "NAD", "NADH", "FADH",
    "DNA", "RNA", "ROS", "NF", "NO", "CO", "pH",
}

# Uppercase gene symbols: PDE11A, SNCA, TP53 etc.
_GENE_PATTERN    = re.compile(r'\b([A-Z][A-Z0-9]{1,})\b')
_CHROMOSOME_RE   = re.compile(r'Chromosome\s+Number\s*:\s*\S+\s*', re.IGNORECASE)
# Question sections
_NETWORK_RE      = re.compile(
    r'Network Definition of the pathway:\s*(.*?)(?=Genes in the pathway:|Given this context|$)',
    re.DOTALL | re.IGNORECASE)
_GENE_LIST_RE    = re.compile(
    r'Genes in the pathway:\s*(.*?)(?=Given this context|$)',
    re.DOTALL | re.IGNORECASE)
_CLOSING_RE      = re.compile(r'(Given this context.*)', re.DOTALL | re.IGNORECASE)
# Network token scanner — finds all word-like tokens in pathway notation
_NETWORK_TOKEN   = re.compile(r'\b([A-Za-z][A-Za-z0-9\-]*)\b')
# Pathway removal (for Stage 5)
_PATHWAY_BLOCK_RE = re.compile(
    r'Network Definition of the pathway:.*?(?=Genes in the pathway:|Given this context|$)',
    re.DOTALL | re.IGNORECASE)
_GENE_LIST_BLOCK_RE = re.compile(
    r'Genes in the pathway:.*?(?=Given this context|$)',
    re.DOTALL | re.IGNORECASE)

# ── Known biology molecule list ───────────────────────────────────────────────
# Each entry: (canonical_name, [alias1, alias2, ...])
# Longer aliases must come first within each group — _apply_mol_map sorts globally.
_KNOWN_MOL_GROUPS: List[Tuple[str, List[str]]] = [
    # Cyclic nucleotides
    ("cAMP",            ["cyclic adenosine monophosphate", "3',5'-cyclic AMP", "cyclic AMP", "cAMP"]),
    ("cGMP",            ["cyclic guanosine monophosphate", "cyclic GMP", "cGMP"]),
    # Dopamine pathway
    ("L-Dopa",          ["L-3,4-dihydroxyphenylalanine", "levodopa", "L-DOPA", "L-Dopa"]),
    ("dopamine",        ["dopaminergic", "dopamine"]),
    ("tyrosine hydroxylase", ["tyrosine hydroxylase"]),
    # Cortisol / adrenal
    ("cortisol",        ["hydrocortisol", "cortisol", "glucocorticoid"]),
    ("ACTH",            ["adrenocorticotropic hormone", "corticotropin", "ACTH"]),
    ("cholesterol",     ["cholesterol"]),
    # Aggregation / neurodegeneration
    ("alpha-synuclein", ["alpha-synuclein", "α-synuclein", "synuclein alpha", "alpha synuclein"]),
    ("Lewy bodies",     ["Lewy bodies", "Lewy body"]),
    ("tau",             ["tau protein", "tau"]),
    ("amyloid-beta",    ["amyloid-beta", "amyloid beta", "beta-amyloid", "Aβ", "amyloid"]),
    ("ubiquitin",       ["ubiquitin"]),
    ("LC3",             ["MAP1LC3", "LC3"]),
    ("p62",             ["sequestosome", "p62"]),
    # Signaling molecules
    ("IP3",             ["inositol-1,4,5-trisphosphate", "inositol trisphosphate", "IP3"]),
    ("DAG",             ["diacylglycerol", "DAG"]),
    ("calcium",         ["Ca2+", "Ca²⁺", "calcium ion", "calcium"]),
    ("ROS",             ["reactive oxygen species", "ROS"]),
    ("nitric oxide",    ["nitric oxide"]),
    ("NF-kB",           ["nuclear factor-kappa B", "nuclear factor kappa B", "NF-κB", "NF-kB"]),
    ("mTOR",            ["mechanistic target of rapamycin", "mTOR"]),
    ("Wnt",             ["Wnt signaling", "Wnt"]),
    ("Notch",           ["Notch signaling", "Notch"]),
    ("Hedgehog",        ["Hedgehog signaling", "Hedgehog"]),
    # Neurotransmitters
    ("serotonin",       ["5-hydroxytryptamine", "5-HT", "serotonin"]),
    ("norepinephrine",  ["noradrenaline", "norepinephrine"]),
    ("epinephrine",     ["adrenaline", "epinephrine"]),
    ("acetylcholine",   ["acetylcholine"]),
    ("glutamate",       ["glutamic acid", "glutamate"]),
    ("GABA",            ["gamma-aminobutyric acid", "GABA"]),
    # Hormones / steroids
    ("testosterone",    ["testosterone"]),
    ("estrogen",        ["oestrogen", "estradiol", "estrogen"]),
    ("progesterone",    ["progesterone"]),
    ("insulin",         ["insulin"]),
    # Nucleotides
    ("ATP",             ["adenosine triphosphate", "ATP"]),
    ("ADP",             ["adenosine diphosphate", "ADP"]),
    ("GTP",             ["guanosine triphosphate", "GTP"]),
    # Enzymes with strong disease hints
    ("phosphodiesterase", ["phosphodiesterase"]),
    ("steroidogenic acute regulatory protein", ["steroidogenic acute regulatory", "STAR protein"]),
]


# ── Gene / molecule helpers ───────────────────────────────────────────────────

def _build_gene_map(text: str) -> Dict[str, str]:
    """Scan text and return {GENE_SYMBOL: GENE_N} in order of first appearance."""
    gene_map: Dict[str, str] = {}
    counter = [1]
    def collect(m: re.Match) -> str:
        name = m.group(1)
        if name not in _NON_GENE_TERMS and name not in gene_map:
            gene_map[name] = f"GENE_{counter[0]}"
            counter[0] += 1
        return m.group(0)
    _GENE_PATTERN.sub(collect, text)
    return gene_map


def _shuffle_gene_map(gene_map: Dict[str, str]) -> Dict[str, str]:
    genes   = list(gene_map.keys())
    indices = list(range(1, len(genes) + 1))
    random.shuffle(indices)
    return {g: f"GENE_{i}" for g, i in zip(genes, indices)}


def build_global_gene_map(all_rows: list) -> Dict[str, str]:
    """
    Build a single gene->GENE_N map across ALL examples (sorted alphabetically
    for reproducibility). Every example uses this same map so the same gene
    always gets the same anonymous token regardless of which example it's in.
    """
    all_genes: set = set()
    for ex in all_rows:
        combined = ex.get("question", "") + " " + ex.get("reasoning", "")
        for m in _GENE_PATTERN.finditer(combined):
            name = m.group(1)
            if name not in _NON_GENE_TERMS:
                all_genes.add(name)
    sorted_genes = sorted(all_genes)
    return {gene: f"GENE_{i+1}" for i, gene in enumerate(sorted_genes)}


def build_global_mol_map() -> Dict[str, str]:
    """
    Build a single mol->MOL_N map from ALL _KNOWN_MOL_GROUPS aliases.
    Every alias for a canonical molecule gets the same MOL_N token, and
    this map is identical across all examples — no per-example variation.
    This pre-assigns every known molecule so _apply_mol_map catches all
    occurrences in both question and reasoning, eliminating leaks.
    """
    mol_map: Dict[str, str] = {}
    lower_to_token: Dict[str, str] = {}
    counter = [1]

    for canonical, aliases in _KNOWN_MOL_GROUPS:
        # Assign a fresh token for this group (longest alias → canonical token)
        token = f"MOL_{counter[0]}"
        counter[0] += 1
        for alias in aliases:
            key = alias.lower()
            if key not in lower_to_token:
                lower_to_token[key] = token
            mol_map[alias] = lower_to_token[key]

    return mol_map


def _apply_gene_map(text: str, gene_map: Dict[str, str]) -> str:
    def replace(m: re.Match) -> str:
        return gene_map.get(m.group(1), m.group(1))
    return _GENE_PATTERN.sub(replace, text)


def _build_mol_map(network_after_gene_anon: str,
                   full_text: str) -> Dict[str, str]:
    """
    Build per-example molecule map from two sources:
    1. All non-GENE_N tokens in the pathway network string
    2. Known biology molecule list matched in full_text (question + reasoning)
    Used only in --shuffle mode; default mode uses the global mol map instead.
    """
    mol_map:  Dict[str, str] = {}
    lower_to_token: Dict[str, str] = {}
    counter = [1]

    def _add(name: str) -> None:
        key = name.lower()
        if key in lower_to_token:
            mol_map[name] = lower_to_token[key]
        else:
            token = f"MOL_{counter[0]}"
            counter[0] += 1
            mol_map[name] = token
            lower_to_token[key] = token

    # ── Source 1: scan pathway network tokens ─────────────────────────────────
    for m in _NETWORK_TOKEN.finditer(network_after_gene_anon):
        token = m.group(1)
        if not token.startswith("GENE_"):
            _add(token)

    # ── Source 2: known biology list — check presence in combined text ────────
    text_lower = full_text.lower()
    for canonical, aliases in _KNOWN_MOL_GROUPS:
        found = any(alias.lower() in text_lower for alias in aliases)
        if found:
            group_token = lower_to_token.get(canonical.lower())
            if group_token is None:
                for alias in aliases:
                    group_token = lower_to_token.get(alias.lower())
                    if group_token is not None:
                        break
            if group_token is None:
                group_token = f"MOL_{counter[0]}"
                counter[0] += 1
            for alias in aliases:
                mol_map[alias] = group_token
                lower_to_token[alias.lower()] = group_token

    return mol_map


def _apply_mol_map(text: str, mol_map: Dict[str, str]) -> str:
    """Replace all molecule names — longest first to avoid partial matches."""
    for mol, token in sorted(mol_map.items(), key=lambda x: -len(x[0])):
        # Use lookarounds instead of \b to handle special chars like 5'-AMP, Ca2+
        pattern = r'(?<![A-Za-z0-9])' + re.escape(mol) + r'(?![A-Za-z0-9])'
        text = re.sub(pattern, token, text, flags=re.IGNORECASE)
    return text


# ── Question structure helpers ────────────────────────────────────────────────

def _extract_gene_descriptions(gene_list_text: str) -> Dict[str, List[str]]:
    """
    Parse 'SNCA; synuclein alpha | TH; tyrosine hydroxylase | ...'
    into {gene_symbol: [description]} for use in protein-name anonymization.
    """
    desc_map: Dict[str, List[str]] = {}
    for entry in gene_list_text.split('|'):
        entry = entry.strip()
        if ';' in entry:
            symbol, desc = entry.split(';', 1)
            desc_map[symbol.strip()] = [d.strip() for d in desc.strip().split(',')]
    return desc_map


def _remove_chromosome(text: str) -> str:
    return _CHROMOSOME_RE.sub('', text).strip()


def _remove_pathway_and_gene_list(text: str) -> str:
    """Remove Network Definition + Genes in the pathway sections."""
    text = _PATHWAY_BLOCK_RE.sub('', text)
    text = _GENE_LIST_BLOCK_RE.sub('', text)
    return text.strip()


# ── Per-example processing ────────────────────────────────────────────────────

@dataclass
class StageConfig:
    name:             str
    description:      str
    keep_chr:         bool
    keep_pathway:     bool
    anon_genes:       bool   # replace gene names with GENE_N
    shuffle_genes:    bool   # randomise GENE_N indices per example (--shuffle mode only)
    anon_molecules:   bool   # replace pathway molecules with MOL_N
    anon_reasoning:   bool   # apply same anonymisation to reasoning
    include_reasoning: bool  # False -> empty reasoning (no CoT in label)


def _process_example(
    ex: dict,
    cfg: StageConfig,
    global_gene_map: Optional[Dict[str, str]] = None,
    global_mol_map:  Optional[Dict[str, str]] = None,
) -> dict:
    question  = ex["question"]
    reasoning = ex["reasoning"]

    gene_map: Dict[str, str] = {}
    mol_map:  Dict[str, str] = {}

    # ── extract pathway network before any modification ───────────────────────
    network_match = _NETWORK_RE.search(question)
    network_text  = network_match.group(1).strip() if network_match else ""

    gene_list_match = _GENE_LIST_RE.search(question)
    gene_list_text  = gene_list_match.group(1).strip() if gene_list_match else ""

    # ── always remove "Genes in the pathway:" section (big leakage) ──────────
    question = _GENE_LIST_BLOCK_RE.sub('', question).strip()

    # ── structural removals ───────────────────────────────────────────────────
    if not cfg.keep_chr:
        question = _remove_chromosome(question)
    if not cfg.keep_pathway:
        question = _PATHWAY_BLOCK_RE.sub('', question).strip()

    # ── gene anonymisation ────────────────────────────────────────────────────
    if cfg.anon_genes:
        if global_gene_map is not None:
            # Global mode: same gene -> same GENE_N across all examples
            gene_map = global_gene_map
        else:
            # Shuffle mode: per-example map, indices randomised
            gene_map = _build_gene_map(question + " " + gene_list_text + " " + reasoning)
            if cfg.shuffle_genes:
                gene_map = _shuffle_gene_map(gene_map)
        question = _apply_gene_map(question, gene_map)
        if cfg.anon_reasoning:
            reasoning = _apply_gene_map(reasoning, gene_map)

    # ── molecule anonymisation ────────────────────────────────────────────────
    if cfg.anon_molecules:
        if global_mol_map is not None:
            # Global mode: pre-assigned MOL_N for ALL known molecules — no leaks
            mol_map = global_mol_map
        else:
            # Shuffle mode: per-example map built from network + known list
            network_anon = _apply_gene_map(network_text, gene_map) if gene_map else network_text
            combined     = question + " " + reasoning
            mol_map      = _build_mol_map(network_anon, combined)

            # Also add protein descriptions from gene list as mol aliases
            desc_map = _extract_gene_descriptions(gene_list_text)
            for symbol, descs in desc_map.items():
                if symbol in gene_map:
                    token = gene_map[symbol]
                    for desc in descs:
                        if desc and desc.lower() not in [k.lower() for k in mol_map]:
                            mol_map[desc] = token

        question  = _apply_mol_map(question,  mol_map)
        if cfg.anon_reasoning:
            reasoning = _apply_mol_map(reasoning, mol_map)

    # ── reasoning inclusion ───────────────────────────────────────────────────
    if not cfg.include_reasoning:
        reasoning = ""

    return {
        "split":              ex.get("split", ""),
        "answer":             ex["answer"],
        "original_question":  ex["question"],
        "original_reasoning": ex["reasoning"],
        "anon_question":      question.strip(),
        "anon_reasoning":     reasoning.strip(),
        "gene_map":           str(gene_map),
        "mol_map":            str({k: v for k, v in mol_map.items() if not k.startswith("GENE_")}),
        "reference_sequence": ex["reference_sequence"],
        "variant_sequence":   ex["variant_sequence"],
    }


FIELDNAMES = [
    "split", "answer",
    "original_question", "original_reasoning",
    "anon_question", "anon_reasoning",
    "gene_map", "mol_map",
    "reference_sequence", "variant_sequence",
]


def _write_csv(rows: list, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved -> {path}  ({len(rows)} rows)")


# ── Stage and test definitions ────────────────────────────────────────────────

STAGES: Dict[int, StageConfig] = {
    0: StageConfig(
        name="stage0_anon_genes_mol_no_chr",
        description="Anon genes+mol, no Chr | Linear",
        keep_chr=False,  keep_pathway=True,
        anon_genes=True, shuffle_genes=False, anon_molecules=True,
        anon_reasoning=True, include_reasoning=True,
    ),
    1: StageConfig(
        name="stage1_anon_genes_mol_keep_chr",
        description="Anon genes+mol + Chr | Linear",
        keep_chr=True,   keep_pathway=True,
        anon_genes=True, shuffle_genes=False, anon_molecules=True,
        anon_reasoning=True, include_reasoning=True,
    ),
    2: StageConfig(
        name="stage2_real_genes_keep_chr",
        description="Real genes + pathway + Chr | migrate Linear -> Cross-attn",
        keep_chr=True,   keep_pathway=True,
        anon_genes=False, shuffle_genes=False, anon_molecules=False,
        anon_reasoning=False, include_reasoning=True,
    ),
    3: StageConfig(
        name="stage3_real_genes_no_chr",
        description="Real genes + pathway, no Chr | Cross-attn",
        keep_chr=False,  keep_pathway=True,
        anon_genes=False, shuffle_genes=False, anon_molecules=False,
        anon_reasoning=False, include_reasoning=True,
    ),
    4: StageConfig(
        name="stage4_anon_question_dna_only",
        description="Anon question + DNA only, no pathway, no Chr | migrate Cross-attn -> Linear",
        keep_chr=False,  keep_pathway=False,
        anon_genes=True, shuffle_genes=False, anon_molecules=False,
        anon_reasoning=False, include_reasoning=True,
    ),
}

TESTS: Dict[int, StageConfig] = {
    2: StageConfig(
        name="test2_anon_keep_chr",
        description="Anon genes+mol, Chr, pathway | Anon reasoning",
        keep_chr=True,  keep_pathway=True,
        anon_genes=True, shuffle_genes=False, anon_molecules=True,
        anon_reasoning=True, include_reasoning=True,
    ),
    3: StageConfig(
        name="test3_anon_no_chr",
        description="Anon genes+mol, no Chr, pathway | Anon reasoning",
        keep_chr=False, keep_pathway=True,
        anon_genes=True, shuffle_genes=False, anon_molecules=True,
        anon_reasoning=True, include_reasoning=True,
    ),
    4: StageConfig(
        name="test4_anon_no_chr_no_pathway",
        description="Anon question + no pathway, no Chr | Anon reasoning",
        keep_chr=False, keep_pathway=False,
        anon_genes=True, shuffle_genes=False, anon_molecules=False,
        anon_reasoning=True, include_reasoning=True,
    ),
    5: StageConfig(
        name="test5_anon_question_dna_only_no_reasoning",
        description="Anon question + DNA only | No reasoning (hardest eval)",
        keep_chr=False, keep_pathway=False,
        anon_genes=True, shuffle_genes=False, anon_molecules=False,
        anon_reasoning=False, include_reasoning=False,
    ),
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate curriculum stage and test CSVs for KEGG training."
    )
    parser.add_argument("--cache_dir", default="~/.cache/huggingface")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--prefix", default="",
                        help="Optional prefix for all output filenames.")
    parser.add_argument("--stages", type=int, nargs="*",
                        help="Which stages to generate (default: all).")
    parser.add_argument("--tests",  type=int, nargs="*",
                        help="Which tests to generate (default: all). "
                             "Test 1 uses HF directly — skip or ignored.")
    parser.add_argument("--output_dir", default=".",
                        help="Directory to write CSV files into.")
    parser.add_argument("--shuffle", action="store_true",
                        help="Per-example shuffled gene/mol indices instead of global consistent map. "
                             "Output filenames get 'shuffle_' prefix instead of 'global_'.")
    args = parser.parse_args()

    random.seed(args.seed)

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # In shuffle mode, enable per-example randomised indices in all configs
    if args.shuffle:
        for cfg in list(STAGES.values()) + list(TESTS.values()):
            cfg.shuffle_genes = True

    # Output filename prefix: "global_" by default, "shuffle_" with --shuffle
    mode_prefix = "shuffle_" if args.shuffle else "global_"
    user_prefix = (args.prefix + "_") if args.prefix else ""
    full_prefix = user_prefix + mode_prefix

    stages_to_run = args.stages if args.stages is not None else list(STAGES.keys())
    tests_to_run  = args.tests  if args.tests  is not None else list(TESTS.keys())

    print("Loading wanglab/kegg …")
    dataset = load_dataset("wanglab/kegg", cache_dir=args.cache_dir)

    all_rows = []
    for split_name, split_data in dataset.items():
        for ex in split_data:
            ex_dict = dict(ex)
            ex_dict["split"] = split_name
            all_rows.append(ex_dict)

    print(f"Loaded {len(all_rows)} examples.\n")

    # ── Build maps ────────────────────────────────────────────────────────────
    global_gene_map: Optional[Dict[str, str]] = None
    global_mol_map:  Optional[Dict[str, str]] = None

    if not args.shuffle:
        global_gene_map = build_global_gene_map(all_rows)
        print(f"Global gene map: {len(global_gene_map)} unique genes.")
        global_mol_map = build_global_mol_map()
        print(f"Global mol map:  {len(global_mol_map)} aliases across {len(_KNOWN_MOL_GROUPS)} molecule groups.\n")
    else:
        print("Shuffle mode: per-example randomised gene/mol indices.\n")

    # ── Curriculum stages ─────────────────────────────────────────────────────
    print("=== Curriculum Stages ===\n")

    for stage_id in stages_to_run:
        if stage_id not in STAGES:
            print(f"  Stage {stage_id} -> not defined, skipping")
            continue

        cfg      = STAGES[stage_id]
        filename = f"{full_prefix}{cfg.name}.csv"
        filepath = os.path.join(args.output_dir, filename)

        print(f"  Stage {stage_id}: {cfg.description}")
        processed = [_process_example(ex, cfg, global_gene_map, global_mol_map)
                     for ex in all_rows]
        _write_csv(processed, filepath)

    # ── Test configurations ───────────────────────────────────────────────────
    print("\n=== Test Configurations ===")
    print("  Test 1  -> use HF dataset directly (wanglab/kegg), no CSV needed.\n")

    for test_id in tests_to_run:
        if test_id == 1:
            print("  Test 1 -> HF dataset (skipped)")
            continue
        if test_id not in TESTS:
            print(f"  Test {test_id} -> not defined, skipping")
            continue

        cfg      = TESTS[test_id]
        filename = f"{full_prefix}{cfg.name}.csv"
        filepath = os.path.join(args.output_dir, filename)

        print(f"  Test {test_id}: {cfg.description}")
        processed = [_process_example(ex, cfg, global_gene_map, global_mol_map)
                     for ex in all_rows]
        _write_csv(processed, filepath)

    print("\nDone. Pass any CSV to training with:")
    print("  --kegg_csv <path_to_csv>")


if __name__ == "__main__":
    main()
