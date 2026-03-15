# Speech BCI: Transformer Experiment Progress

**Last Updated:** 2026-03-15

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
| `NeuralDecoder/neuralDecoder/neuralSequenceDecoder.py` | Added `TransformerEncoder` instantiation (with `gradientCheckpointing` passthrough); patience-based early stopping (`earlyStopPatience`, `earlyStopMinDelta`); dtype-safe normalization layer (`tf.cast` for mixed precision) |
| `NeuralDecoder/neuralDecoder/configs/config.yaml` | Added `earlyStopPatience: 0`, `earlyStopMinDelta: 0.0`, and `mixedPrecision: false` defaults |
| `NeuralDecoder/neuralDecoder/configs/model/transformer_stack_inputNet.yaml` | New config file for the Transformer model (includes `gradientCheckpointing: false` default) |
| `NeuralDecoder/neuralDecoder/main.py` | Fixed 20GB GPU memory pool; mixed precision activation (`tf.keras.mixed_precision.set_global_policy('mixed_float16')` when `mixedPrecision=true`) |
| `setup_runpod.sh` | Full environment setup for RunPod RTX 4090 |
| `AnalysisExamples/run_round1_experiments.py` | Round 1 runner (16 configs × 1k batches); OOM auto-retry (up to 3×); stale error.log cleanup on each attempt |
| `AnalysisExamples/run_round2_experiments.py` | Round 2 runner (top 8 × 5k batches); same OOM retry and cleanup |
| `AnalysisExamples/run_round3_experiments.py` | Round 3 runner (top 4 × 20k batches); same OOM retry and cleanup; mixed precision + gradient checkpointing enabled |

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

## 5. Known Issues

### VRAM OOM on Long Training Runs
The Transformer's attention mechanism is O(T²) in memory (T = sequence length, up to 500 timesteps). Over many training steps, the BFC allocator's memory pool becomes internally fragmented. A worst-case batch where all sequences are near max length then fails to get a contiguous allocation block.

**Fix 1:** Fixed 20GB pre-allocated pool in `main.py` (reduces fragmentation vs. `memory_growth=True`).

**Fix 2:** Gradient checkpointing via `tf.recompute_grad`. When enabled, each `TransformerEncoderLayer`'s activations are not stored during forward pass but recomputed during backprop. This cuts activation memory from O(L×T²) to O(T²) with zero accuracy impact, at the cost of ~33% slower training. Enable via `model.gradientCheckpointing=true`.

**Fix 3:** Mixed precision (`mixedPrecision=true`). Uses float16 for all activations (~2× memory reduction). Weights stay float32. The output Dense layer is forced to float32 for CTC loss stability. Gives a ~1.5–2× training speedup on RTX 4090 Tensor Cores as a bonus. Enable via `mixedPrecision=true` in the Hydra config.

**Fix 4:** OOM auto-retry in the runner scripts. When a training run crashes with `RESOURCE_EXHAUSTED`, the runner automatically resumes from the last checkpoint (up to 3 retries). The allocator state resets on each new process, giving a clean pool. No accuracy impact since optimizer state and weights are fully restored from the checkpoint.

---

## 6. What To Do Next

### 6a. Full Training of Top 2 Configs (Immediate Next Step)
Train both winning architectures for a full production run to find the definitive best model:
- **`transformer_256d_4L_8H_512ff`**: d_model=256, 4 layers, 8 heads, d_ff=512
- **`transformer_512d_4L_8H_2048ff`**: d_model=512, 4 layers, 8 heads, d_ff=2048
- Use the same **19 sessions** (matching Willett et al. baseline for fair comparison)
- Train for longer (e.g., 50k–100k batches) until convergence with early stopping
- Enable gradient checkpointing + mixed precision to avoid OOM
- Compare final PER against the GRU baseline from Willett et al.

### ~~6b. Implement Gradient Checkpointing~~ (DONE)
### ~~6c. Implement Mixed Precision~~ (DONE)
### ~~6d. Complete Architecture Search (Successive Halving)~~ (DONE)

### 6e. End-to-End Model (Thesis Contribution 2)
Once the phoneme decoder is finalized, integrate it with a language model:
- The original Willett et al. uses a 5-gram language model with beam search (code in `LanguageModelDecoder/`)
- The previous thesis (13521081) experimented with Transformer language models
- Goal: wire the Transformer phoneme decoder output into the language model decoding pipeline and evaluate WER (word error rate) on the full vocabulary task

### 6f. Baseline Comparison
Report metrics against:
1. The original GRU decoder (Willett et al. baseline) — run it using `NeuralDecoder/neuralDecoder/configs/model/gru_stack_inputNet.yaml`
2. The previous thesis results (13521081)

---

## 7. Useful Commands

```bash
# New pod setup
bash setup_runpod.sh

# TensorBoard monitoring
tensorboard --logdir=/workspace/speechBCI/experiments/round3 --host=0.0.0.0 --port=6006

# Full training run — winner 1 (256d, 8 heads)
python -m neuralDecoder.main \
    model=transformer_stack_inputNet \
    dataset=speech_release_baseline \
    model.d_model=256 model.num_layers=4 model.nhead=8 model.d_ff=512 \
    model.dropout=0.1 model.posEncType=sinusoidal \
    model.gradientCheckpointing=true \
    mixedPrecision=true \
    outputDir=experiments/final_model_256d_8H \
    gpuNumber=0 \
    nBatchesToTrain=100000 batchesPerVal=500 batchSize=32 \
    learnRateStart=0.0005 learnRateEnd=0.0 learnRateDecaySteps=100000 \
    warmUpSteps=500 gradClipValue=10 lossType=ctc \
    smoothInputs=1 smoothKernelSD=2 \
    earlyStopPatience=20 earlyStopMinDelta=0.0001

# Full training run — winner 2 (512d, 8 heads)
python -m neuralDecoder.main \
    model=transformer_stack_inputNet \
    dataset=speech_release_baseline \
    model.d_model=512 model.num_layers=4 model.nhead=8 model.d_ff=2048 \
    model.dropout=0.1 model.posEncType=sinusoidal \
    model.gradientCheckpointing=true \
    mixedPrecision=true \
    outputDir=experiments/final_model_512d_8H \
    gpuNumber=0 \
    nBatchesToTrain=100000 batchesPerVal=500 batchSize=32 \
    learnRateStart=0.0005 learnRateEnd=0.0 learnRateDecaySteps=100000 \
    warmUpSteps=500 gradClipValue=10 lossType=ctc \
    smoothInputs=1 smoothKernelSD=2 \
    earlyStopPatience=20 earlyStopMinDelta=0.0001
```
