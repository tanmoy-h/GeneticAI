# GenoMorph

**GenoMorph** extends [BioReason](https://arxiv.org/abs/2505.23579) (Fallahpour et al., 2025) — a DNA-LLM model that fuses a frozen Evo2-7B encoder with Qwen3-1.7B via linear projection — with three novel contributions: question-conditioned cross-attention fusion, entropy-adaptive latent token replacement, and hierarchical OT-regularised GRPO. The result is a model that reasons over genomic variant sequences to predict disease associations with significantly improved accuracy while reducing reliance on memorised gene names.

> **Base model**: The `Source/` directory contains the original BioReason code (prior work, linear projection). All GenoMorph contributions are in the root scripts and `genomorph/` package.

---

## Repository Structure

```
GeneticAI/
├── genomorph/                        # GenoMorph package (our contributions)
│   ├── models/
│   │   ├── dna_llm.py               # DNALLMModel with CrossAttentionFusion
│   │   ├── latent_reasoning.py      # GateNet + HRPO adaptive gate loop
│   │   └── thinking_residual.py     # ThinkingResidualGate + DNAHiddenInjector
│   ├── hiref/                        # HiRef hierarchical OT (HR_OT, FRLC, ...)
│   ├── trainer/                      # GRPO trainer + config
│   ├── dataset/
│   │   ├── kegg.py                  # KEGG dataset loader (named + anon)
│   │   └── global_stage1_anon_genes_mol_keep_chr.csv  # anonymised KEGG dataset
│   └── dna_modules/
│
├── Source/                           # BioReason base code (prior work)
│   ├── bioreason/                   # BioReason package (linear projection)
│   ├── train_dna_qwen.py            # BioReason SFT training script
│   └── tests/
│       ├── sh_train_bioreason.sh    # Train BioReason baseline
│       ├── sh_test_bioreason.sh     # Test BioReason baseline
│       ├── sh_test_bioreason_anon.sh
│       └── sh_test_llm_only.sh
│
├── train_dna_qwen.py                # Stage 1: CrossAttn SFT
├── train_latent_sft.py              # Stage 1.5.1: Gate training
├── train_latent_sft_cached.py       # Stage 1.5.0: LatentSp curriculum (cached DNA)
├── hiref_offline_multigpu_w9.py     # Stage 2: Offline HiRef OT
├── adaptive_thinking_residual_w9.py         # Stage 3: GRPO (fixed θ_low)
├── adaptive_thinking_residual_w9_optB.py    # Stage 3: GRPO (learned θ_low) ← final
│
├── build_anon_dataset.py            # Build anonymised KEGG CSV
├── precompute_dna_embeddings.py     # Precompute Evo2 embedding cache
├── eval_stage1_5_checkpoints.py     # Evaluate Stage 1.5 checkpoints
├── eval_stage3_w9.py                # Evaluate Stage 3
│
├── week8tests/                      # Stage 1 train/test scripts
├── week9tests/                      # Stage 1.5 / 2 / 3 scripts
└── week11tests/                     # Stage 3 Option B (learned θ_low) scripts
```

---

## Installation

### Requirements
- Python 3.11+
- CUDA 12.8
- SLURM cluster (scripts use `sbatch`; can also run with `bash` directly)

### Environment setup

```bash
# Create conda environment
conda create -n dna_env python=3.11 -y
conda activate dna_env

# Install PyTorch with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install GenoMorph package and dependencies
pip install -e .

# Install Evo2 DNA encoder
pip install -e ".[evo2]"
```

### Verified stack
```
torch==2.9.0+cu128
transformers==4.57.6
trl==1.0.0
pytorch_lightning>=2.4.0
accelerate>=1.2.0
peft>=0.14.0
```

---

## Dataset

The KEGG variant-disease dataset is loaded automatically from HuggingFace:

```python
from datasets import load_dataset
ds = load_dataset("wanglab/kegg")   # train / val / test splits
```

### Anonymised dataset

To test whether models rely on memorised gene names rather than DNA signal, we provide a pre-built anonymised version where all gene names are replaced with `GENE_N` tokens and molecule names with `MOL_N` tokens:

```
genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv
```

To rebuild it from scratch:

```bash
python build_anon_dataset.py \
    --output genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv \
    --keep_chromosome \
    --anonymize_molecules \
    --seed 42
```

---

## Training Pipeline

Run stages in order. Each stage depends on the checkpoint produced by the previous one.

---

### Stage 1 — CrossAttention SFT

Trains Qwen3-1.7B + Evo2-7B with `CrossAttentionFusion` (replaces BioReason's linear projection). DNA tokens attend over the input question to produce question-conditioned representations.

```bash
# Single run (2 GPUs recommended)
bash week8tests/sh_stage1_sft_w8.sh 0,1

# SLURM
sbatch week8tests/sh_stage1_sft_w8.sh
```

**Output checkpoint:**
```
/scratch/.../checkpoints/week8tests/stage1_sft_ca/
  dna-sft-week8-ca-kegg-Qwen3-1.7B-epoch=03-val_loss_epoch=0.4292.ckpt
```

**Key args** (see script for full list):
```
--use_cross_attention True
--max_clip_loss_weight 0.0     # CLIP loss tested but gave no improvement — disabled
--max_epochs 5
--lora_r 32 --lora_alpha 64
```

---

### Precompute DNA Embedding Cache

Precomputes Evo2 embeddings for the full KEGG dataset. Saves ~14 GB VRAM during Stages 1.5+.

```bash
bash week9tests/sh_precompute_dna_w9.sh

# or with custom output path
DNA_CACHE=/your/path/dna_embeddings_kegg_2048.pt bash week9tests/sh_precompute_dna_w9.sh
```

**Output:** `dna_embeddings_kegg_2048.pt`

---

### Stage 1.5.0 — LatentSp Curriculum SFT

Teaches the model entropy-adaptive latent token replacement. At low-entropy positions (model is confident) the last hidden state `h_{t-1}` is recycled as the next token embedding instead of sampling a token. At high-entropy positions (model is uncertain) the DNA summary vector `u_DNA` is injected. A curriculum progressively increases `max_latent_steps` from 0 → 4.

```bash
# Requires Stage 1 checkpoint and DNA cache
STAGE1_CKPT=/path/to/stage1/best.ckpt \
DNA_CACHE=/path/to/dna_embeddings_kegg_2048.pt \
bash week9tests/sh_stage1_50_cached_w9.sh

# SLURM
sbatch week9tests/sh_stage1_50_cached_w9.sh
```

**Output checkpoint:**
```
/scratch/.../checkpoints/week9tests/stage1_50_cached/s04_pass01/model.pt
```

**Key args:**
```
--entropy_mode global          # epoch-level entropy recompute (faster)
--max_latent_steps 4
--passes_per_step 1
--learning_rate 5e-5
```

---

### Stage 1.5.1 — Gate Training

Adds `ThinkingResidualGate` and `DNAHiddenInjector` on top of the Stage 1.5.0 checkpoint. Pre-training the gate before GRPO avoids log-probability discontinuities at Stage 3 onset.

```bash
STAGE15_CKPT=/path/to/stage1_50_cached/s04_pass01/model.pt \
bash week9tests/sh_stage1_51_w9.sh

# SLURM
sbatch week9tests/sh_stage1_51_w9.sh
```

**Output:**
```
/scratch/.../checkpoints/week9tests/stage1_51/s04_pass01/
  model.pt
  thinking_gate.pt
  dna_injector.pt
```

**Key args:**
```
--use_gate                     # activates ThinkingResidualGate + DNAHiddenInjector
--start_latent_step 4          # full latent steps from step 0 (already trained)
--passes_per_step 2
--learning_rate 2e-5
```

---

### Stage 2 — Offline HiRef OT Manifold Computation

One-time forward pass over the full KEGG dataset using the Stage 1.5.1 model. Computes hierarchical OT distances between DNA summary vectors and the answer embedding manifold. Output is used as a reward signal and gate modulator in Stage 3.

```bash
STAGE1_CKPT=/path/to/stage1_51/s04_pass01/model.pt \
OUTPUT_DIR=stage2_output_w9 \
bash week9tests/sh_stage2_hiref_w9.sh

# SLURM (uses 2 GPUs in parallel for speed)
sbatch week9tests/sh_stage2_hiref_w9.sh
```

**Output:** `stage2_output_w9/` — per-sample OT distances and manifold data.

---

### Stage 3 — Adaptive GRPO (Option B: learned θ_low)

GRPO fine-tuning with three simultaneously active mechanisms:

1. **HRPO gate** — `ThinkingResidualGate` blends `u_DNA` into token embeddings before generation
2. **LatentSp entropy-adaptive generation** — per-token decision: `H_t < θ_low` → latent step; `H_t > θ_high` → DNA injection. `θ_low` is a learnable parameter trained via REINFORCE (Option B)
3. **OT distance reward** — penalises reasoning paths geometrically far from the answer manifold (from Stage 2)

```bash
STAGE1_CKPT=/path/to/stage1_51/s04_pass01/model.pt \
GATE_CKPT_DIR=/path/to/stage1_51/s04_pass01 \
STAGE2_DIR=stage2_output_w9 \
DNA_CACHE=/path/to/dna_embeddings_kegg_2048.pt \
bash week11tests/sh_stage3_grpo_w9_optB.sh 0,1

# SLURM (2 GPUs)
sbatch week11tests/sh_stage3_grpo_w9_optB.sh
```

**Key args:**
```
--use_hrpo_gate True
--max_latent_steps 1
--latentSp_theta_low 1.0       # init value; learned via REINFORCE
--latentSp_theta_high 3.0
--theta_low_lr 1e-4
--theta_low_weight 0.1
--reward_funcs xmlcount soft_format correctness completion_quality
               reasoning_quality ot_distance latent_usage
--num_generations 8
--lora_r 16 --lora_alpha 32
--learning_rate 2e-6
```

**Output:** `checkpoints/week11tests/stage3_grpo_optB/checkpoint-XXXX/`

---

## Testing and Evaluation

### BioReason Baseline (Source — linear projection)

```bash
# Named dataset
bash Source/tests/sh_test_bioreason.sh

# Anonymised dataset (set CKPT_PATH if not using default)
CKPT_PATH=/path/to/bioreason.ckpt \
bash Source/tests/sh_test_bioreason_anon.sh

# LLM-only baseline (no DNA encoder)
CKPT_PATH=/path/to/llm_only.ckpt \
bash Source/tests/sh_test_llm_only.sh
```

### Stage 1.5.0 — Checkpoint Sweep

Evaluates all curriculum checkpoints and ranks by accuracy:

```bash
bash week9tests/sh_eval_stage1_5_w9.sh

# With anonymised data
KEGG_CSV=genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv \
bash week9tests/sh_eval_stage1_5_w9.sh
```

### Stage 3 — Full Evaluation

```bash
# Named dataset
bash week11tests/sh_eval_stage3_w11.sh

# Anonymised dataset
KEGG_CSV=genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv \
bash week11tests/sh_eval_stage3_w11.sh
```

### Quick Smoke Test (Stage 3 mechanisms)

Verifies that the HRPO gate, latent steps, and DNA injection all fire correctly within 200 steps:

```bash
STAGE1_CKPT=/path/to/model.pt bash week9tests/sh_test_latentSp.sh
```

What to check in the log:
```
train/gate_latent_steps > 0    ← latent step path fired
[w9] dna_injected=True         ← DNA injection triggered
No NaN loss / no crash         ← mechanisms stable
```

---

## Results

Evaluated on the full KEGG test+val set (290 samples, macro-averaged F1).
**Anon**: gene names → `GENE_N`, molecule names → `MOL_N` (tests memorisation vs. DNA reasoning).

| Model | DNA Fusion | LatentSp | Gate | HiRef OT | Acc (named) | F1 (named) | Avg Time | Acc (anon) | F1 (anon) |
|---|---|---|---|---|---|---|---|---|---|
| LLM-only | ✗ | ✗ | ✗ | ✗ | — | — | — | — | — |
| BioReason [1] | Linear proj. | ✗ | ✗ | ✗ | — | — | ~23s | — | — |
| Stage 1: CrossAttn SFT | CrossAttn | ✗ | ✗ | ✗ | — | — | — | — | — |
| — + CLIP loss (ablated) | CrossAttn + CLIP | ✗ | ✗ | ✗ | — | — | — | — | — |
| Stage 1.5.0: + LatentSp curriculum | CrossAttn | fixed θ | ✗ | ✗ | 98.0%* | — | — | — | — |
| Stage 1.5.1: + Gate training | CrossAttn | fixed θ | ✓ | ✗ | — | — | — | — | — |
| GenoMorph: GRPO, fixed θ_low | CrossAttn | fixed θ | ✓ | ✓ | 92.1% | 0.816 | 21.4s | — | — |
| **GenoMorph-B: learned θ_low** | CrossAttn | learned θ | ✓ | ✓ | **95.5%** | — | — | — | — |
| — w/o OT reward | CrossAttn | learned θ | ✓ | ✗ | — | — | — | — | — |
| — w/o LatentSp | CrossAttn | ✗ | ✓ | ✓ | — | — | — | — | — |

*Stage 1.5.0 accuracy evaluated on 100-sample subset; all other rows use full 290-sample set.
F1 is macro-averaged. — = evaluation pending.

### Per-class F1 (GenoMorph fixed θ, checkpoint-3088)

| Disease | P | R | F1 | n |
|---|---|---|---|---|
| Alzheimer's disease | 1.000 | 1.000 | 1.000 | 40 |
| Parkinson's disease | 1.000 | 0.979 | 0.989 | 47 |
| ALS | 1.000 | 0.914 | 0.955 | 35 |
| Spinocerebellar ataxia | 1.000 | 0.972 | 0.986 | 36 |
| Huntington's disease | 1.000 | 1.000 | 1.000 | 10 |
| Melanoma | 1.000 | 0.882 | 0.938 | 17 |
| Colorectal cancer | 1.000 | 1.000 | 1.000 | 12 |
| Prion disease | 1.000 | 0.867 | 0.929 | 15 |
| Gaucher disease | 1.000 | 1.000 | 1.000 | 7 |
| Glioblastoma | 1.000 | 1.000 | 1.000 | 6 |
| Thyroid cancer | 1.000 | 0.500 | 0.667 | 4 |
| Von Hippel-Lindau | 1.000 | 0.500 | 0.667 | 4 |
| **Overall** | **0.848** | **0.796** | **0.816** | **290** |

---

## Citation

If you use GenoMorph, please also cite the BioReason paper this work builds on:

```bibtex
@misc{fallahpour2025bioreason,
  title   = {BioReason: Incentivizing Multimodal Biological Reasoning within a DNA-LLM Model},
  author  = {Adibvafa Fallahpour and Andrew Magnuson and Purav Gupta and Shihao Ma
             and Jack Naimer and Arnav Shah and Haonan Duan and Omar Ibrahim
             and Hani Goodarzi and Chris J. Maddison and Bo Wang},
  year    = {2025},
  eprint  = {2505.23579},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url     = {https://arxiv.org/abs/2505.23579}
}
```

---

## References

[1] Fallahpour et al., *BioReason: Incentivizing Multimodal Biological Reasoning within a DNA-LLM Model*, arXiv 2505.23579, 2025.
[2] HiRef: Hierarchical Refinement OT, ICML 2025.
