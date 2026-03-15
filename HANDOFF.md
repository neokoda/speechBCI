# Speech BCI: Transformer Experiment Progress

**Last Updated:** March 2026

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
| `NeuralDecoder/neuralDecoder/models.py` | Added `TransformerEncoder` model class |
| `NeuralDecoder/neuralDecoder/neuralSequenceDecoder.py` | Added `TransformerEncoder` instantiation; added patience-based early stopping (`earlyStopPatience`, `earlyStopMinDelta`) to `train()` |
| `NeuralDecoder/neuralDecoder/configs/config.yaml` | Added `earlyStopPatience: 0` and `earlyStopMinDelta: 0.0` defaults |
| `NeuralDecoder/neuralDecoder/configs/model/transformer_stack_inputNet.yaml` | New config file for the Transformer model |
| `NeuralDecoder/neuralDecoder/main.py` | Fixed 20GB GPU memory pool (replaces `memory_growth=True`) |
| `setup_runpod.sh` | Full environment setup for RunPod RTX 4090 |
| `AnalysisExamples/run_round1_experiments.py` | Round 1 experiment runner (16 configs × 1k batches) |
| `AnalysisExamples/run_round2_experiments.py` | Round 2 experiment runner (top 8 × 5k batches) |
| `AnalysisExamples/run_round3_experiments.py` | Round 3 experiment runner (top 4 × 20k batches) |

---

## 4. Architecture Search: Successive Halving (COMPLETE)

We ran a 3-round Successive Halving search over Transformer hyperparameters (`d_model`, `num_layers`, `nhead`, `d_ff`). Each round doubles the training budget and halves the number of candidates.

| Round | Configs | Batches each | Promoted |
|---|---|---|---|
| Round 1 | 16 | 1,000 | Top 8 |
| Round 2 | 8 | 5,000 | Top 4 |
| Round 3 | 4 | 20,000 | Top 2 (final) |

### Final Results (Round 3)

| Rank | Config | d_model | layers | heads | d_ff | R3 PER |
|---|---|---|---|---|---|---|
| 🥇 | transformer_256d_4L_4H_1024ff | 256 | 4 | 4 | 1024 | **0.4974** |
| 🥈 | transformer_512d_4L_4H_2048ff | 512 | 4 | 4 | 2048 | 0.5010 |
| 3 | transformer_512d_4L_4H_1024ff | 512 | 4 | 4 | 1024 | 0.5052 |
| — | transformer_256d_4L_8H_512ff | 256 | 4 | 8 | 512 | FAIL (OOM) |

**Winner: `transformer_256d_4L_4H_1024ff`** (d_model=256, 4 layers, 4 heads, d_ff=1024)

Notable finding: the 512d models dominated in the shorter rounds but were overtaken by the smaller 256d model at the full 20k-batch budget. The smaller model generalizes better given enough training time.

Results are saved in:
- `experiments/round3/final_results.json`
- `experiments/round3/round3_results.csv`

### Data Notes
- Training uses **19 of the 24 available sessions** (as defined in `NeuralDecoder/neuralDecoder/configs/dataset/speech_release_baseline.yaml`), totalling ~6,640 train sentences and ~680 val sentences.
- 5 sessions exist on disk but are unused: `t12.2022.06.23`, `t12.2022.07.29`, `t12.2022.08.18`, `t12.2022.08.23`, `t12.2022.08.25` (~2,160 extra sentences). These were excluded from the original Willett et al. baseline for comparability.
- The `test` split in the tfRecords is used as the validation set throughout training. There is no separate held-out test set in the released data.

---

## 5. Known Issues

### VRAM OOM on Long Training Runs
The Transformer's attention mechanism is O(T²) in memory (T = sequence length, up to 500 timesteps). Over many training steps, the BFC allocator's memory pool becomes internally fragmented. A worst-case batch where all sequences are near max length then fails to get a contiguous allocation block.

**Current fix:** Fixed 20GB pre-allocated pool in `main.py` (reduces fragmentation vs. `memory_growth=True`).

**Better fix (not yet implemented):** Gradient checkpointing. Instead of keeping all layer activations in memory for backprop, recompute them layer-by-layer during the backward pass. This cuts activation memory from O(L×T²) to O(T²) with zero accuracy impact, at the cost of ~33% slower training.

---

## 6. What To Do Next

### 6a. Final Model Training (Immediate Next Step)
Train the winning architecture (`transformer_256d_4L_4H_1024ff`) for a full production run:
- Use **all 24 sessions** (not just 19) for maximum data
- Train for longer (e.g., 50k–100k batches) until convergence
- Implement gradient checkpointing first to avoid OOM on longer runs
- Compare final PER against the GRU baseline from Willett et al.

### 6b. Implement Gradient Checkpointing
Before the final long training run, add gradient checkpointing to `TransformerEncoder` in `NeuralDecoder/neuralDecoder/models.py`. In TF/Keras this is done via `tf.recompute_grad` on the layer call. This will prevent OOM crashes on very long training runs without affecting model quality.

### 6c. End-to-End Model (Thesis Contribution 2)
Once the phoneme decoder is finalized, integrate it with a language model:
- The original Willett et al. uses a 5-gram language model with beam search (code in `LanguageModelDecoder/`)
- The previous thesis (13521081) experimented with Transformer language models
- Goal: wire the Transformer phoneme decoder output into the language model decoding pipeline and evaluate WER (word error rate) on the full vocabulary task

### 6d. Baseline Comparison
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

# Run the winning config for a full training run
python -m neuralDecoder.main \
    model=transformer_stack_inputNet \
    dataset=speech_release_baseline \
    model.d_model=256 model.num_layers=4 model.nhead=4 model.d_ff=1024 \
    model.dropout=0.1 model.posEncType=sinusoidal \
    outputDir=experiments/final_model \
    gpuNumber=0 \
    nBatchesToTrain=50000 batchesPerVal=500 batchSize=32 \
    learnRateStart=0.0005 learnRateEnd=0.0 learnRateDecaySteps=50000 \
    warmUpSteps=500 gradClipValue=10 lossType=ctc \
    smoothInputs=1 smoothKernelSD=2 \
    earlyStopPatience=20 earlyStopMinDelta=0.0001
```
