# GenoMorph — Ablation Results

Metric columns (`Acc`, `Avg Time`, `F1`, named + anon) are filled by running each
row's eval script. Configuration columns are fixed by the scripts below.

- **Named** scripts: `train/`, `test/` (BioReason: `src/scripts/real/`).
- **Anon** scripts: `train/anon/`, `test/anon/` (BioReason: `src/scripts/anon/`, `_anon` suffix).
- The four Stage 3 rows share the same eval wrapper; only `CKPT=` differs. Two
  wrappers exist: `test_06b_eval_stage3.sh` (one-directional score) and
  `test_06b_eval_stage3_final.sh` (bidirectional score) — see **Scoring** below.
- Stage 1.5.0/1.5.1 have a full-sweep eval (ranks all checkpoints) and a
  best-only eval (`test_04c`/`test_04d`, auto-selects and scores just the best
  checkpoint).
- All evals report the **290-record** number (val+test = `--split both` /
  `EVAL_SPLIT=both`; the val split alone is 144 for both named and anon).
- Stage 3 checkpoints are written to each training script's `OUTPUT_DIR` under
  `CK=/scratch/tanmoyh_iitp/GenoMorph/checkpoints`. Set `CKPT=$CK/<dir>/checkpoint-<best-step>`
  (pick the best-`eval_correctness` step, e.g. from the `[KeepBestN]` log line).

### Scoring

Two answer-matching rules (`is_correct`), reported side by side for Stage 3:

- **One-directional** (`gt in pred`) — strict; ground truth must appear in the
  prediction. Used by `eval_stage1_5_checkpoints.py`, `eval_stage1_51_checkpoints.py`,
  `eval_grpo_checkpoint.py` (→ `test_06b_eval_stage3.sh`). This is the primary,
  consistent metric across all rows.
- **Bidirectional** (`gt in pred OR pred in gt`) — lenient. Used by
  `eval_grpo_checkpoint_final.py` (→ `test_06b_eval_stage3_final.sh`), provided as
  a comparison variant for the Stage 3 GRPO rows only.

## Configuration + metrics

| Model | DNA Fusion | LatentSp | Latent@infer | Gate | HiRef OT | Acc (named) | F1 (named) | Avg Time (named) | Acc (anon) | F1 (anon) | Avg Time (anon) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LLM-only | ✗ | ✗ | — | ✗ | ✗ | — | — | — | — | — | — |
| BioReason (LLM + DNA) | Linear proj. | ✗ | — | ✗ | ✗ | — | — | — | — | — | — |
| Stage 1: CrossAttn SFT | CrossAttn | ✗ | ✗ | ✗ | ✗ | — | — | — | — | — | — |
| — + CLIP (ablated) | CrossAttn + CLIP | ✗ | ✗ | ✗ | ✗ | — | — | — | — | — | — |
| Stage 1.5.0: + LatentSp curriculum | CrossAttn | fixed θ | ✗ | ✗ | ✗ | — | — | — | — | — | — |
| Stage 1.5.1: + Gate training | CrossAttn | fixed θ | ✗ | ✓ | ✗ | — | — | — | — | — | — |
| GenoMorph: GRPO, fixed θ_low=0 | CrossAttn | fixed θ | ✗ | ✓ | ✓ | — | — | — | — | — | — |
| GenoMorph: GRPO, fixed θ_low=1 | CrossAttn | fixed θ | ✓ | ✓ | ✓ | — | — | — | — | — | — |
| **GenoMorph-B: learned θ_low** | CrossAttn | learned θ | ✓ | ✓ | ✓ | — | — | — | — | — | — |
| — w/o OT reward | CrossAttn | learned θ | ✓ | ✓ | ✗ | — | — | — | — | — | — |
| — w/o LatentSp | CrossAttn | ✗ | ✗ | ✓ | ✓ | — | — | — | — | — | — |

Column meanings:
- **LatentSp** — latent-space reasoning **trained into the model**: `fixed θ` =
  fixed-threshold curriculum (Stages 1.5.0/1.5.1), `learned θ` = GRPO-learned
  threshold (Stage 3), `✗` = never trained with the latent curriculum.
- **Latent@infer** — whether latent reasoning steps **actually fire in the
  reported eval**. The Stage 1.5.0/1.5.1 evals run **text-only** (latent tokens
  banned, and no autonomous "when-to-fire" controller exists until Stage 3), so
  latents don't contribute to those numbers *even though the curriculum is
  trained in*. Stage 3 fires latents via `generate_with_hrpo_gate`. (`—` = model
  has no latent capability at all.)
- **Gate** — `ThinkingResidualGate` (+ `DNAHiddenInjector`); active in the 1.5.1
  eval too (forward is patched even though latent tokens are banned).

`Acc`/`F1` above are the **one-directional** score. For the four Stage 3 GRPO
rows, also record the **bidirectional** score from the `_final` scripts (e.g. as
`Acc 1-dir / bi-dir`) — see **Scoring** above.

## Scripts (named + anon)

| Model | Train (named) | Eval (named) | Train (anon) | Eval (anon) |
|---|---|---|---|---|
| LLM-only | `train/train_01_llm_only.sh` | `test/test_01_llm_only.sh` | `train/anon/train_01_llm_only.sh` | `test/anon/test_01_llm_only.sh` |
| BioReason (LLM + DNA) | `src/scripts/real/sh_train_bioreason_sft.sh` | `src/scripts/real/sh_test_bioreason_sft.sh` | `src/scripts/anon/sh_train_bioreason_sft_anon.sh` | `src/scripts/anon/sh_test_bioreason_sft_anon.sh` |
| Stage 1: CrossAttn SFT | `train/train_02_stage1_sft.sh` | `test/test_02_stage1.sh` | `train/anon/train_02_stage1_sft.sh` | `test/anon/test_02_stage1.sh` |
| — + CLIP (ablated) | `train/train_02b_stage1_clip.sh` | `test/test_02_stage1.sh` | `train/anon/train_02b_stage1_clip.sh` | `test/anon/test_02_stage1.sh` |
| Stage 1.5.0: + LatentSp curriculum | `train/train_03b_stage1_50_cached.sh` | sweep: `test/test_04_eval_stage1_5.sh`<br>best: `test/test_04c_eval_stage1_5_best.sh` | `train/anon/train_03b_stage1_50_cached.sh` | sweep: `test/anon/test_04_eval_stage1_5.sh`<br>best: `test/anon/test_04c_eval_stage1_5_best.sh` |
| Stage 1.5.1: + Gate training | `train/train_04_stage1_51.sh` | sweep: `test/test_04b_eval_stage1_51.sh`<br>best: `test/test_04d_eval_stage1_51_best.sh` | `train/anon/train_04_stage1_51.sh` | sweep: `test/anon/test_04b_eval_stage1_51.sh`<br>best: `test/anon/test_04d_eval_stage1_51_best.sh` |
| GenoMorph: GRPO, fixed θ_low=0 | `train/train_06_stage3_grpo.sh` | 1-dir: `THETA_LOW=0 THETA_LOW_PT= CKPT=$CK/train_06_stage3_grpo/checkpoint-N bash test/test_06b_eval_stage3.sh`<br>bi-dir: same envs+`CKPT=` + `test/test_06b_eval_stage3_final.sh` | `train/anon/train_06_stage3_grpo.sh` | 1-dir: `THETA_LOW=0 THETA_LOW_PT= CKPT=$CK/train_06_stage3_grpo_anon/checkpoint-N bash test/anon/test_06b_eval_stage3.sh`<br>bi-dir: same envs+`CKPT=` + `test/anon/test_06b_eval_stage3_final.sh` |
| GenoMorph: GRPO, fixed θ_low=1 | `train/train_06_stage3_grpo.sh` | 1-dir: `THETA_LOW=1 THETA_LOW_PT= CKPT=$CK/train_06_stage3_grpo/checkpoint-N bash test/test_06b_eval_stage3.sh`<br>bi-dir: same envs+`CKPT=` + `test/test_06b_eval_stage3_final.sh` | `train/anon/train_06_stage3_grpo.sh` | 1-dir: `THETA_LOW=1 THETA_LOW_PT= CKPT=$CK/train_06_stage3_grpo_anon/checkpoint-N bash test/anon/test_06b_eval_stage3.sh`<br>bi-dir: same envs+`CKPT=` + `test/anon/test_06b_eval_stage3_final.sh` |
| **GenoMorph-B: learned θ_low** | `train/train_06b_stage3_grpo_optB.sh` | 1-dir: `CKPT=$CK/train_06b_stage3_grpo_optB/checkpoint-N bash test/test_06b_eval_stage3.sh`<br>bi-dir: same `CKPT=` + `test/test_06b_eval_stage3_final.sh` | `train/anon/train_06b_stage3_grpo_optB.sh` | 1-dir: `CKPT=$CK/train_06b_stage3_grpo_optB_anon/checkpoint-N bash test/anon/test_06b_eval_stage3.sh`<br>bi-dir: same `CKPT=` + `test/anon/test_06b_eval_stage3_final.sh` |
| — w/o OT reward | `train/train_06c_stage3_grpo_no_ot.sh` | 1-dir: `CKPT=$CK/stage3_grpo_no_ot/checkpoint-N bash test/test_06b_eval_stage3.sh`<br>bi-dir: same `CKPT=` + `test/test_06b_eval_stage3_final.sh` | `train/anon/train_06c_stage3_grpo_no_ot.sh` | 1-dir: `CKPT=$CK/stage3_grpo_no_ot_anon/checkpoint-N bash test/anon/test_06b_eval_stage3.sh`<br>bi-dir: same `CKPT=` + `test/anon/test_06b_eval_stage3_final.sh` |
| — w/o LatentSp | `train/train_06d_stage3_grpo_no_latentsp.sh` | 1-dir: `CKPT=$CK/stage3_grpo_no_latentsp/checkpoint-N bash test/test_06b_eval_stage3.sh`<br>bi-dir: same `CKPT=` + `test/test_06b_eval_stage3_final.sh` | `train/anon/train_06d_stage3_grpo_no_latentsp.sh` | 1-dir: `CKPT=$CK/stage3_grpo_no_latentsp_anon/checkpoint-N bash test/anon/test_06b_eval_stage3.sh`<br>bi-dir: same `CKPT=` + `test/anon/test_06b_eval_stage3_final.sh` |

### Inference-only latent ablation (alternative to retraining `06d`)

Scores the **GenoMorph-B** checkpoint with latents forced off, no training:

- Named: `CKPT=$CK/train_06b_stage3_grpo_optB/checkpoint-N MODE=ablation bash train/train_06e_eval_latent_ablation.sh`
- Anon:  `CKPT=$CK/train_06b_stage3_grpo_optB_anon/checkpoint-N MODE=ablation bash train/anon/train_06e_eval_latent_ablation.sh`

`MODE=both` runs baseline (latents ON) + ablation (latents OFF) on the same
checkpoint; the delta in `eval_correctness` is the latent-reasoning contribution.
