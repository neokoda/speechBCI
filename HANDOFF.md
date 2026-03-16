# Speech BCI: Transformer Experiment Progress

**Last Updated:** 2026-03-16

This document is the primary handoff for anyone resuming work on this thesis project. It covers the long-term goal, what has been built and completed so far, the current state of the codebase, and what needs to be done next.

---

## 1. Thesis Goal

This repo is the codebase for an undergraduate thesis (TA / Tugas Akhir). The starting point is the original Willett et al. (2023) Speech BCI repository, which implements an RNN (GRU)-based phoneme decoder for a Brain-Computer Interface that decodes attempted speech from intracortical electrode recordings.

**The thesis has two main contributions:**

1. **Transformer Phoneme Decoder:** Replace the original GRU model with a Transformer encoder as the neural sequence decoder, and find the best Transformer architecture via a systematic hyperparameter search (Successive Halving). The goal is to match or beat the GRU's phoneme error rate (PER).

2. **End-to-End Model:** After the best Transformer decoder is found, integrate it with a language model (as the original paper does) to build a full speech-to-text pipeline. The previous thesis (13521081) implemented this with an n-gram and Transformer language model; this thesis aims to improve or extend that.

**Key reference papers:**
- `s41586-023-06377-x.pdf` — Willett et al. (2023), Nature. The original paper this repo implements.
- `laporanTugasAkhir-13521081-FINALFINAL.docx.pdf` — Previous thesis that implemented a Transformer language model on top of the same RNN decoder. This is the direct predecessor work.
- `13522108-ProposalTA-signed.pdf` — The proposal for this thesis.

---

## 2. Environment Setup (RunPod)

This project runs on a **RunPod RTX 4090** instance. On every new pod, run the setup script first:

```bash
bash setup_runpod.sh
```

This script:
1. Installs `tensorflow==2.15.0.post1` and pinned NVIDIA pip packages (`nvidia-cudnn-cu12==8.9.7.29`, etc.)
2. Sets `LD_LIBRARY_PATH` in `~/.bashrc` to point to the NVIDIA pip libraries (required for GPU detection)
3. Installs the `NeuralDecoder` package in editable mode (`pip install -e NeuralDecoder/`)
4. Runs a smoke test to verify TF detects the GPU and `TransformerEncoder` works

**Why these specific versions?** The default RunPod PyTorch template ships with CUDA 12.4. TF 2.15.0.post1 + cudnn 8.9.7.29 is the combination that works correctly on CUDA 12.4 hardware without requiring a system-level CUDA install.

**GPU memory:** `NeuralDecoder/neuralDecoder/main.py` sets a **fixed 20GB memory pool** (via `set_logical_device_configuration`) instead of using `memory_growth=True`. Memory growth mode causes BFC allocator fragmentation over long training runs — the pool fills up with non-contiguous free blocks, and a single long-sequence batch that needs a large contiguous attention matrix then crashes. The fixed pool gives the allocator one large contiguous slab to manage.

---

## 3. Codebase Modifications

All modifications from the original Willett et al. repo:

| File | Change |
|---|---|
| `NeuralDecoder/neuralDecoder/models.py` | Added `TransformerEncoder` model class with gradient checkpointing (`tf.recompute_grad`); output Dense layer forced to `dtype='float32'` for CTC stability under mixed precision; dtype-safe scaling and positional encoding addition |
| `NeuralDecoder/neuralDecoder/neuralSequenceDecoder.py` | Added `TransformerEncoder` instantiation (with `gradientCheckpointing` passthrough); patience-based early stopping (`earlyStopPatience`, `earlyStopMinDelta`); cosine annealing LR schedule (`lrScheduleType: cosine`); dtype-safe normalization layer (`tf.cast` for mixed precision) |
| `NeuralDecoder/neuralDecoder/configs/config.yaml` | Added `earlyStopPatience: 0`, `earlyStopMinDelta: 0.0`, `mixedPrecision: false`, and `lrScheduleType: polynomial` defaults |
| `NeuralDecoder/neuralDecoder/configs/model/transformer_stack_inputNet.yaml` | New config file for the Transformer model (includes `gradientCheckpointing: false` default) |
| `NeuralDecoder/neuralDecoder/main.py` | Fixed 20GB GPU memory pool; mixed precision activation (`tf.keras.mixed_precision.set_global_policy('mixed_float16')` when `mixedPrecision=true`) |
| `setup_runpod.sh` | Full environment setup for RunPod RTX 4090 |
| `AnalysisExamples/run_round1_experiments.py` | Round 1 runner (16 configs × 1k batches); OOM auto-retry (up to 3×); stale error.log cleanup on each attempt |
| `AnalysisExamples/run_round2_experiments.py` | Round 2 runner (top 8 × 5k batches); same OOM retry and cleanup |
| `AnalysisExamples/run_round3_experiments.py` | Round 3 runner (top 4 × 20k batches); same OOM retry and cleanup; mixed precision + gradient checkpointing enabled |
| `AnalysisExamples/run_final_training.py` | Final training runner (top 2 × 100k batches); OOM auto-retry (up to 10×); early stopping patience=20 |
| `AnalysisExamples/run_cosine_ablation.py` | Cosine annealing ablation (256d model × 200k batches); LR=0.001, cosine decay to 0.0001, patience=30 |

---

## 4. Architecture Search: Successive Halving (COMPLETE)

We ran a 3-round Successive Halving search over Transformer hyperparameters (`d_model`, `num_layers`, `nhead`, `d_ff`). Each round doubles the training budget and halves the number of candidates.

All three rounds used **19 sessions** matching Willett et al. (2023) for fair comparison. Round 2 was re-run after the original run used batchSize=64 which caused 4 of 8 configs to OOM; the re-run used batchSize=32, LR=0.0005 (linear scaling rule), and gradient checkpointing uniformly. This changed the Round 3 promotion set.

| Round | Configs | Batches each | Settings | Promoted |
|---|---|---|---|---|
| Round 1 | 16 | 1,000 | bs=64, lr=0.001 | Top 8 |
| Round 2 | 8 | 5,000 | bs=32, lr=0.0005, grad ckpt | Top 4 |
| Round 3 | 4 | 20,000 | bs=32, lr=0.0005, grad ckpt, mixed precision | Top 2 |

### Final Results (Round 3)

| Rank | Config | d_model | layers | heads | d_ff | R3 PER |
|---|---|---|---|---|---|---|
| 🥇 | transformer_256d_4L_8H_512ff | 256 | 4 | 8 | 512 | **0.4984** |
| 🥈 | transformer_512d_4L_8H_2048ff | 512 | 4 | 8 | 2048 | 0.5012 |
| 3 | transformer_512d_4L_4H_2048ff | 512 | 4 | 4 | 2048 | 0.5037 |
| 4 | transformer_512d_6L_4H_1024ff | 512 | 6 | 4 | 1024 | 0.5052 |

**Top 2 winners: `transformer_256d_4L_8H_512ff` and `transformer_512d_4L_8H_2048ff`**

Notable findings:
- All top 4 configs use 4 layers. Deeper models (6 layers) consistently underperformed at this training budget.
- The top 2 use 8 heads; they OOM'd initially but completed after auto-retry from checkpoint with a fresh allocator pool.
- Both winners have relatively small d_ff (512 and 2048 relative to d_model), suggesting the FFN bottleneck is not the limiting factor.

Results are saved in:
- `experiments/round3/final_results.json`
- `experiments/round3/round3_results.csv`

### Data Notes
- Training uses **19 of the 24 available sessions** (as defined in `NeuralDecoder/neuralDecoder/configs/dataset/speech_release_baseline.yaml`), totalling ~6,640 train sentences and ~680 val sentences. This matches the original Willett et al. (2023) paper, which used 19 sessions for its reported results.
- 5 sessions exist on disk but are unused: `t12.2022.06.23`, `t12.2022.07.29`, `t12.2022.08.18`, `t12.2022.08.23`, `t12.2022.08.25` (~2,160 extra sentences). These were excluded from the original Willett et al. baseline. Using 19 sessions ensures a fair comparison against the GRU baseline.
- The `test` split in the tfRecords is used as the validation set throughout training. There is no separate held-out test set in the released data.

---

## 5. Full Training Results (COMPLETE)

### 5a. Linear Decay Training

Both winners trained for 100k batches with linear LR decay (0.0005 → 0.0), early stopping patience=20.

Script: `AnalysisExamples/run_final_training.py`

| Model | Best PER | At Step | Status |
|---|---|---|---|
| **transformer_256d_4L_8H_512ff** | **0.3671** | ~93,500 | Completed 100k |
| transformer_512d_4L_8H_2048ff | 0.3826 | ~100,000 | Completed 100k (patience 19/20) |

The 256d model is the clear winner. The gap widened with more training (0.003 at 20k → 0.016 at 100k), confirming the architecture search ranking.

Results: `experiments/final_training/final_training_results.json`

### 5b. Cosine Annealing Ablation

The linear decay models were LR-limited in the tail (LR near 0 in the last 20% of training, yet PER still improving). We added cosine annealing support (`lrScheduleType: cosine` in config) and ran the 256d model with a more aggressive schedule:

- **Peak LR: 0.001** (2× higher), cosine decay to **min LR: 0.0001** (floor, never starves)
- **200k steps** (2× longer), **warmup: 1000 steps**
- **Early stopping patience: 30**

Script: `AnalysisExamples/run_cosine_ablation.py`

| Run | Best PER | At Step | Status |
|---|---|---|---|
| Linear decay (baseline) | 0.3671 | ~93,500 | Completed 100k |
| **Cosine annealing** | **0.3351** | 118,500 | Early stopped at 133.5k |

**8.7% relative improvement** over linear decay. The cosine schedule keeps LR productive for longer — at step 118.5k the LR was still ~0.0004 (vs near-zero for linear at the same point). The model early-stopped after 30 val cycles (15k steps) of no improvement at a healthy LR, suggesting it reached the architecture's capacity rather than being LR-starved.

Results: `experiments/cosine_ablation/cosine_ablation_result.json`

### 5c. Hyperparameter Notes

**Adam epsilon=0.1**: The optimizer in `neuralSequenceDecoder.py` uses `epsilon=1e-01` (line 224), which is unusually high (standard is 1e-8). This was inherited from the original Willett et al. GRU codebase. It makes Adam more conservative (behaves like SGD+momentum when gradients are small). All architecture search and training results use this value — do NOT change without a separate ablation, as it would invalidate all comparisons.

**OOM retries**: Long training runs (>50k batches) on the 256d model typically OOM 4-6 times due to BFC allocator fragmentation from variable-length attention batches. The runner scripts handle this with auto-retry from checkpoint (up to 10 retries). Each retry gives a fresh allocator pool.

---

## 6. Known Issues

### VRAM OOM on Long Training Runs
The Transformer's attention mechanism is O(T²) in memory (T = sequence length, up to 500 timesteps). Over many training steps, the BFC allocator's memory pool becomes internally fragmented. A worst-case batch where all sequences are near max length then fails to get a contiguous allocation block.

**Fix 1:** Fixed 20GB pre-allocated pool in `main.py` (reduces fragmentation vs. `memory_growth=True`).

**Fix 2:** Gradient checkpointing via `tf.recompute_grad`. When enabled, each `TransformerEncoderLayer`'s activations are not stored during forward pass but recomputed during backprop. This cuts activation memory from O(L×T²) to O(T²) with zero accuracy impact, at the cost of ~33% slower training. Enable via `model.gradientCheckpointing=true`.

**Fix 3:** Mixed precision (`mixedPrecision=true`). Uses float16 for all activations (~2× memory reduction). Weights stay float32. The output Dense layer is forced to float32 for CTC loss stability. Gives a ~1.5–2× training speedup on RTX 4090 Tensor Cores as a bonus. Enable via `mixedPrecision=true` in the Hydra config.

**Fix 4:** OOM auto-retry in the runner scripts. When a training run crashes with `RESOURCE_EXHAUSTED`, the runner automatically resumes from the last checkpoint (up to 10 retries). The allocator state resets on each new process, giving a clean pool. No accuracy impact since optimizer state and weights are fully restored from the checkpoint.

---

## 7. What To Do Next

### ~~7a. Full Training of Top 2 Configs~~ (DONE — see Section 5a)
### ~~7b. Implement Gradient Checkpointing~~ (DONE)
### ~~7c. Implement Mixed Precision~~ (DONE)
### ~~7d. Complete Architecture Search (Successive Halving)~~ (DONE)
### ~~7e. Cosine Annealing Ablation~~ (DONE — see Section 5b)

### 7f. GRU Baseline Comparison (Next Step)
Train the original GRU model with the same 19 sessions and evaluation to get a fair PER comparison. The existing `AnalysisExamples/rnn_step2_trainBaselineRNN.py` uses different settings from the original Willett et al. paper (1024 units, kernel_size=32, LR=0.02, 10k batches) vs the yaml config defaults (512 units, kernel_size=14). Need to clarify which settings represent the "true" GRU baseline before running.

**Important**: The GRU's raw PER (CTC greedy decode) has never been measured on our 19-session setup. The notebook `rnn_step3_baselineRNNInference.ipynb` only reports **WER after language model decoding (18.9%)**, which is not comparable to our raw Transformer PER of 0.3351.

**Options:**
1. Train GRU with original Willett settings (1024 units, kernel_size=32, LR=0.02) — matches the paper
2. Train GRU with yaml defaults (512 units, kernel_size=14) — matches the config file
3. Run both and report the better one as the baseline

A runner script `AnalysisExamples/run_gru_baseline.py` was started but not finalized. It uses the yaml defaults (option 2). Consider updating to match the original paper settings (option 1) for a stronger baseline.

### 7g. End-to-End Model (Thesis Contribution 2)
Once the phoneme decoder is finalized, integrate it with a language model:
- The original Willett et al. uses a 5-gram language model with beam search (code in `LanguageModelDecoder/`)
- The previous thesis (13521081) experimented with Transformer language models
- Goal: wire the Transformer phoneme decoder output into the language model decoding pipeline and evaluate WER (word error rate) on the full vocabulary task

### 7h. Further Improvements (Optional Ablations)
If time permits, the following could potentially improve the Transformer PER further:
1. **Attention dropout** (`model.attentionDropout`): Currently 0.0. The wiring exists in `models.py`. Setting to 0.05 could help regularization.
2. **Adam epsilon ablation**: Test standard epsilon=1e-8 vs the current 0.1. Could unlock faster convergence but risky — may destabilize training.
3. **SpecAugment / data augmentation**: Currently only white noise + constant offset. Adding time/frequency masking requires code changes to the data pipeline.

---

## 8. Useful Commands

```bash
# New pod setup
bash setup_runpod.sh

# TensorBoard monitoring
tensorboard --logdir=/workspace/speechBCI/experiments --host=0.0.0.0 --port=6006

# Final training (linear decay, both configs)
python AnalysisExamples/run_final_training.py \
    --data-dir /workspace/speechBCI/data/derived/tfRecords \
    --output-dir /workspace/speechBCI/experiments/final_training \
    --gpu 0

# Cosine annealing ablation (256d model only)
python AnalysisExamples/run_cosine_ablation.py \
    --data-dir /workspace/speechBCI/data/derived/tfRecords \
    --output-dir /workspace/speechBCI/experiments/cosine_ablation \
    --gpu 0

# GRU baseline (TODO — script needs to be finalized)
python AnalysisExamples/run_gru_baseline.py \
    --data-dir /workspace/speechBCI/data/derived/tfRecords \
    --output-dir /workspace/speechBCI/experiments/gru_baseline \
    --gpu 0
```

---

## 9. Summary of Best Results

| Model | Schedule | Best PER | Notes |
|---|---|---|---|
| **Transformer 256d_4L_8H_512ff** | **Cosine (0.001→0.0001)** | **0.3351** | **Best result. Early stopped at 133.5k steps.** |
| Transformer 256d_4L_8H_512ff | Linear (0.0005→0.0) | 0.3671 | LR-limited in the tail |
| Transformer 512d_4L_8H_2048ff | Linear (0.0005→0.0) | 0.3826 | Larger model, worse result |
| GRU baseline (Willett et al.) | — | **TBD** | Raw PER not yet measured; WER after LM = 18.9% |
