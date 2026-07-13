# GenoMorph — Ablation Results

Metric columns (`Acc`, `Avg Time`, `F1`, named + anon) are filled by running each
row's eval script. Configuration columns are fixed by the scripts below.

- **Named** scripts: `train/`, `test/` (BioReason: `src/scripts/real/`).
- **Anon** scripts: `train/anon/`, `test/anon/` (BioReason: `src/scripts/anon/`, `_anon` suffix).
- The four Stage 3 rows share `test_06b_eval_stage3.sh`; only `CKPT=` differs.
- Stage 3 checkpoints are written to each training script's `OUTPUT_DIR` under
  `CK=/scratch/tanmoyh_iitp/GenoMorph/checkpoints`. Set `CKPT=$CK/<dir>/checkpoint-<best-step>`
  (pick the best-`eval_correctness` step, e.g. from the `[KeepBestN]` log line).

## Configuration + metrics

| Model | DNA Fusion | LatentSp | Gate | HiRef OT | Acc (named) | Avg Time | F1 (named) | Acc (anon) | F1 (anon) |
|---|---|---|---|---|---|---|---|---|---|
| LLM-only | ✗ | ✗ | ✗ | ✗ | — | — | — | — | — |
| BioReason (LLM + DNA) | Linear proj. | ✗ | ✗ | ✗ | — | — | — | — | — |
| Stage 1: CrossAttn SFT | CrossAttn | ✗ | ✗ | ✗ | — | — | — | — | — |
| — + CLIP (ablated) | CrossAttn + CLIP | ✗ | ✗ | ✗ | — | — | — | — | — |
| Stage 1.5.0: + LatentSp curriculum | CrossAttn | fixed θ | ✗ | ✗ | — | — | — | — | — |
| Stage 1.5.1: + Gate training | CrossAttn | fixed θ | ✓ | ✗ | — | — | — | — | — |
| GenoMorph: GRPO, fixed θ_low | CrossAttn | fixed θ | ✓ | ✓ | — | — | — | — | — |
| **GenoMorph-B: learned θ_low** | CrossAttn | learned θ | ✓ | ✓ | — | — | — | — | — |
| — w/o OT reward | CrossAttn | learned θ | ✓ | ✗ | — | — | — | — | — |
| — w/o LatentSp | CrossAttn | ✗ | ✓ | ✓ | — | — | — | — | — |

## Scripts (named + anon)

| Model | Train (named) | Eval (named) | Train (anon) | Eval (anon) |
|---|---|---|---|---|
| LLM-only | `train/train_01_llm_only.sh` | `test/test_01_llm_only.sh` | `train/anon/train_01_llm_only.sh` | `test/anon/test_01_llm_only.sh` |
| BioReason (LLM + DNA) | `src/scripts/real/sh_train_bioreason_sft.sh` | `src/scripts/real/sh_test_bioreason_sft.sh` | `src/scripts/anon/sh_train_bioreason_sft_anon.sh` | `src/scripts/anon/sh_test_bioreason_sft_anon.sh` |
| Stage 1: CrossAttn SFT | `train/train_02_stage1_sft.sh` | `test/test_02_stage1.sh` | `train/anon/train_02_stage1_sft.sh` | `test/anon/test_02_stage1.sh` |
| — + CLIP (ablated) | `train/train_02b_stage1_clip.sh` | `test/test_02_stage1.sh` | `train/anon/train_02b_stage1_clip.sh` | `test/anon/test_02_stage1.sh` |
| Stage 1.5.0: + LatentSp curriculum | `train/train_03b_stage1_50_cached.sh` | `test/test_04_eval_stage1_5.sh` | `train/anon/train_03b_stage1_50_cached.sh` | `test/anon/test_04_eval_stage1_5.sh` |
| Stage 1.5.1: + Gate training | `train/train_04_stage1_51.sh` | `test/test_04b_eval_stage1_51.sh` | `train/anon/train_04_stage1_51.sh` | `test/anon/test_04b_eval_stage1_51.sh` |
| GenoMorph: GRPO, fixed θ_low | `train/train_06_stage3_grpo.sh` | `CKPT=$CK/train_06_stage3_grpo/checkpoint-N bash test/test_06b_eval_stage3.sh` | `train/anon/train_06_stage3_grpo.sh` | `CKPT=$CK/train_06_stage3_grpo_anon/checkpoint-N bash test/anon/test_06b_eval_stage3.sh` |
| **GenoMorph-B: learned θ_low** | `train/train_06b_stage3_grpo_optB.sh` | `CKPT=$CK/train_06b_stage3_grpo_optB/checkpoint-N bash test/test_06b_eval_stage3.sh` | `train/anon/train_06b_stage3_grpo_optB.sh` | `CKPT=$CK/train_06b_stage3_grpo_optB_anon/checkpoint-N bash test/anon/test_06b_eval_stage3.sh` |
| — w/o OT reward | `train/train_06c_stage3_grpo_no_ot.sh` | `CKPT=$CK/stage3_grpo_no_ot/checkpoint-N bash test/test_06b_eval_stage3.sh` | `train/anon/train_06c_stage3_grpo_no_ot.sh` | `CKPT=$CK/stage3_grpo_no_ot_anon/checkpoint-N bash test/anon/test_06b_eval_stage3.sh` |
| — w/o LatentSp | `train/train_06d_stage3_grpo_no_latentsp.sh` | `CKPT=$CK/stage3_grpo_no_latentsp/checkpoint-N bash test/test_06b_eval_stage3.sh` | `train/anon/train_06d_stage3_grpo_no_latentsp.sh` | `CKPT=$CK/stage3_grpo_no_latentsp_anon/checkpoint-N bash test/anon/test_06b_eval_stage3.sh` |

### Inference-only latent ablation (alternative to retraining `06d`)

Scores the **GenoMorph-B** checkpoint with latents forced off, no training:

- Named: `CKPT=$CK/train_06b_stage3_grpo_optB/checkpoint-N MODE=ablation bash train/train_06e_eval_latent_ablation.sh`
- Anon:  `CKPT=$CK/train_06b_stage3_grpo_optB_anon/checkpoint-N MODE=ablation bash train/anon/train_06e_eval_latent_ablation.sh`

`MODE=both` runs baseline (latents ON) + ablation (latents OFF) on the same
checkpoint; the delta in `eval_correctness` is the latent-reasoning contribution.
