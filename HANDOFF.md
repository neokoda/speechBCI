# Speech BCI: Transformer Experiment Progress

**Last Updated:** March 2026

This document serves as a "handoff" for developers to quickly understand the current state of the architecture search experiments, what has been completed, and what the immediate next steps are.

## 1. Project Context
The goal is to optimize a Transformer-based Neural Sequence Decoder for a Speech Brain-Computer Interface (BCI). We are running successive halving experiments to find the best Transformer hyperparameters (`d_model`, `num_layers`, `nhead`, `d_ff`) by evaluating validation sequence error rate (PER).

## 2. What Has Been Completed

### 2a. Early Stopping Implementation
Early stopping was previously missing. We added patience-based early stopping to the core training loop:
- **Modified:** `NeuralDecoder/neuralDecoder/neuralSequenceDecoder.py` (added `earlyStopPatience`, `earlyStopMinDelta` tracker in `train()`).
- **Modified:** `NeuralDecoder/neuralDecoder/configs/config.yaml` (added default `earlyStopPatience: 0` so it doesn't break baseline runs).

### 2b. Round 2 Experiment Runner
We created a new script to take the top 8 configs from Round 1 and train them for 5,000 batches each, utilizing the new early stopping feature.
- **Created:** `AnalysisExamples/run_round2_experiments.py`
- Inherits tracking logic from Round 1 but trains 5x longer, evaluates twice as often, and promotes the top 4 configs to Round 3.

### 2c. RunPod GPU Environment Fixes
Running TensorFlow on modern RTX 4090 RunPods (which use CUDA 12.4 by default) resulted in GPU detection failures and DNN initialization crashes.
- **Modified:** `setup_runpod.sh`. Upgraded to `tensorflow==2.15.0.post1` and explicitly pinned NVIDIA pip packages (`nvidia-cudnn-cu12`, etc.). Added `LD_LIBRARY_PATH` exports to `~/.bashrc`.
- **Modified:** `NeuralDecoder/neuralDecoder/main.py`. Added `tf.config.experimental.set_memory_growth(gpu, True)`. This prevents TF 2.15 from forcefully allocating all 24GB VRAM upfront, which was causing the fatal `DNN library initialization failed` error.
- **Result:** The RTX 4090 is now correctly utilized for training, and TensorBoard logs (`events.out.tfevents`) write successfully.

## 3. Current State
- The codebase is fully prepared for Round 2.
- The GPU is verified strictly working with TensorFlow 2.15.
- The user manually halted the background run of Round 2 and intends to run it later.

## 4. What To Do Next
1. **Run Round 2:**
   Execute the round 2 script on the RunPod:
   ```bash
   python AnalysisExamples/run_round2_experiments.py \
       --data-dir /workspace/speechBCI/data/derived/tfRecords \
       --output-dir /workspace/speechBCI/experiments/round2 \
       --round1-dir /workspace/speechBCI/experiments/round1 \
       --gpu 0
   ```
2. **Monitor:** Use `tensorboard --logdir=/workspace/speechBCI/experiments/round2 --host=0.0.0.0 --port=6006` to watch the early stopping dynamics.
3. **Round 3 (Future):** Once Round 2 finishes, it will generate `/workspace/speechBCI/experiments/round2/promoted_to_round3.json`. A new script (`run_round3_experiments.py`) will need to be created to take the top 4 configs and train them for even longer (e.g., 20,000 batches).
