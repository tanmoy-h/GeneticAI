"""
Thin wrapper around train_grpo.py that supports any HF dataset via --dataset_name.

The original train_grpo.py hardcodes 'wanglab/kegg' inside get_kegg_questions()
and never reads script_args.dataset_name.  This wrapper patches that function
before main() is called so any uploaded anonymized dataset can be used.

Usage (same flags as train_grpo.py, but --dataset_name is now honoured):
    python train_grpo_anon.py \
        --dataset_name iitp-cse/kegg-anon-stage1 \
        --sft_checkpoint /path/to/sft.ckpt \
        ... (all other train_grpo.py flags)
"""

import os
import sys

# ── patch before train_grpo is fully initialised ──────────────────────────────
import train_grpo as _grpo
from datasets import load_dataset
from bioreason.dataset.utils import truncate_dna
from bioreason.dataset.kegg import format_kegg_for_dna_llm


def _make_patched_loader(dataset_name: str):
    def _patched(truncate_dna_per_side: int = 0):
        data = load_dataset(dataset_name)
        if truncate_dna_per_side > 0:
            data = data.map(truncate_dna, fn_kwargs={"truncate_dna_per_side": truncate_dna_per_side})
        data = data.map(format_kegg_for_dna_llm, fn_kwargs={"is_sft": False})
        return data
    return _patched


if __name__ == "__main__":
    # Read --dataset_name from argv before TrlParser consumes it
    dataset_name = "wanglab/kegg"
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--dataset_name" and i + 1 < len(sys.argv):
            dataset_name = sys.argv[i + 1]
            break

    # Patch the module-level function so main() uses the correct dataset
    _grpo.get_kegg_questions = _make_patched_loader(dataset_name)

    # Run original __main__ logic unchanged
    os.environ.setdefault("HF_DATASETS_DISABLE_MULTIPROCESSING", "1")
    os.environ.setdefault("WANDB_PROJECT", "dna-grpo")

    from trl import TrlParser
    from train_grpo import GRPOScriptArguments, DNALLMGRPOConfig, GRPOModelConfig, main

    parser = TrlParser((GRPOScriptArguments, DNALLMGRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
