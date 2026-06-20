# Rename Log

## Python Files

| Old Name | New Name | Reason |
|----------|----------|--------|
| `adaptive_thinking_residual_w9.py` | `train_grpo_latent_reasoning.py` | GRPO training with entropy-conditioned dual-mode reasoning (HRPO gate + latent skip steps + DNA hidden injection) |
| `adaptive_thinking_residual_w9_optB.py` | `train_grpo_learned_theta.py` | Same as above but `theta_low` is a learnable `nn.Parameter` trained via REINFORCE instead of fixed |
| `eval_stage3_w9.py` | `eval_grpo_checkpoint.py` | Evaluates GRPO checkpoints — greedy generation, accuracy/precision/recall/F1, per-class metrics, saves `.json` + `.csv` |
| `hiref_offline_multigpu_w9.py` | `hiref_kegg_align.py` | Stage 2 offline HiRef alignment over full KEGG dataset, multi-GPU sharded extract + merge |

## Shell Script Folders (week → train/test)

### Root level

| Old Name | New Name |
|----------|----------|
| `week9tests/sh_precompute_dna_w9.sh` | `train/train_01_precompute_dna.sh` |
| `week8tests/sh_stage1_sft_w8.sh` | `train/train_02_stage1_sft.sh` |
| `week9tests/sh_stage1_50_w9.sh` | `train/train_03_stage1_50.sh` |
| `week9tests/sh_stage1_50_cached_w9.sh` | `train/train_04_stage1_50_cached.sh` |
| `week9tests/sh_stage1_51_w9.sh` | `train/train_05_stage1_51.sh` |
| `week9tests/sh_stage2_hiref_w9.sh` | `train/train_06_stage2_hiref.sh` |
| `week9tests/sh_stage3_grpo_w9.sh` | `train/train_07_stage3_grpo.sh` |
| `week11tests/sh_stage3_grpo_w9_optB.sh` | `train/train_08_stage3_grpo_optB.sh` |
| `week8tests/sh_test_stage1_w8.sh` | `test/test_01_stage1.sh` |
| `week9tests/sh_test_stage1_5_w9.sh` | `test/test_02_stage1_5.sh` |
| `week9tests/sh_eval_stage1_5_w9.sh` | `test/test_03_eval_stage1_5.sh` |
| `week9tests/sh_test_latentSp.sh` | `test/test_04_latentSp.sh` |
| `week9tests/sh_eval_stage3_w9.sh` | `test/test_05_eval_stage3.sh` |
| `week11tests/sh_test_optB_quick.sh` | `test/test_06_optB_quick.sh` |
| `week11tests/sh_eval_stage3_w11.sh` | `test/test_07_eval_stage3_w11.sh` |
| `week11tests/test_theta_low_reinforce.py` | `test/test_08_theta_low_reinforce.py` |

## Folder Rename

| Old Name | New Name |
|----------|----------|
| `src/` | `src/` |

### src folder (formerly Source)

| Old Name | New Name |
|----------|----------|
| `src/sh_train_sft_anon.sh` | `src/train/train_01_sft_anon.sh` |
| `src/sh_grpo.sh` | `src/train/train_02_grpo.sh` |
| `src/sh_train_grpo_anon.sh` | `src/train/train_03_grpo_anon.sh` |
| `src/sh_train_dna_only.sh` | `src/train/train_04_dna_only.sh` |
| `src/sh_train_dna_qwen.sh` | `src/train/train_05_dna_qwen.sh` |
| `src/sh_convert_deepspeed_to_hf_ckpt_dna.sh` | `src/train/train_06_convert_deepspeed.sh` |
| `src/sh_convert_grpo_to_hf_ckpt.sh` | `src/train/train_07_convert_grpo.sh` |
| `src/tests/sh_train_bioreason.sh` | `src/train/train_08_bioreason.sh` |
| `src/tests/sh_train_bioreason_anon.sh` | `src/train/train_09_bioreason_anon.sh` |
| `src/tests/sh_train_llm_only.sh` | `src/train/train_10_llm_only.sh` |
| `src/sh_test_anon.sh` | `src/test/test_01_anon.sh` |
| `src/tests/sh_test_bioreason.sh` | `src/test/test_02_bioreason.sh` |
| `src/tests/sh_test_bioreason_anon.sh` | `src/test/test_03_bioreason_anon.sh` |
| `src/tests/sh_test_llm_only.sh` | `src/test/test_04_llm_only.sh` |

## Shell Scripts Updated (py file references)

| Script | Reference updated |
|--------|------------------|
| `train/train_06_stage2_hiref.sh` | `hiref_offline_multigpu_w9.py` → `hiref_kegg_align.py` |
| `train/train_07_stage3_grpo.sh` | `adaptive_thinking_residual_w9.py` → `train_grpo_latent_reasoning.py` |
| `train/train_08_stage3_grpo_optB.sh` | `adaptive_thinking_residual_w9_optB.py` → `train_grpo_learned_theta.py` |
| `test/test_04_latentSp.sh` | `adaptive_thinking_residual_w9.py` → `train_grpo_latent_reasoning.py` |
| `test/test_05_eval_stage3.sh` | `eval_stage3_w9.py` → `eval_grpo_checkpoint.py` |
| `test/test_06_optB_quick.sh` | `adaptive_thinking_residual_w9_optB.py` → `train_grpo_learned_theta.py` |
| `test/test_07_eval_stage3_w11.sh` | `eval_stage3_w9.py` → `eval_grpo_checkpoint.py` |
