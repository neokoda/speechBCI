# Speech BCI: Transformer Experiment Progress

**Last Updated:** 2026-04-06 (Session 9 — WFST LM Pipeline)

Primary handoff for resuming thesis work. Covers goals, completed work, current state, and next steps.

---

## 1. Thesis Goal

Starting point: Willett et al. (2023) Speech BCI repo — GRU-based phoneme decoder for intracortical electrode recordings.

**Two thesis contributions:**

1. **Transformer Phoneme Decoder:** Replace GRU with Transformer encoder; find best architecture via Successive Halving. **COMPLETE** — Conformer 24-sess PER=0.1654 beats GRU PER=0.1818.
2. **Full Speech Pipeline:** Integrate best decoder with a language model. **IN PROGRESS** — WFST 3-gram gives WER=0.222; need 5-gram to approach Willett's 0.118.

**Key references:**
- `s41586-023-06377-x.pdf` — Willett et al. (2023) original paper
- `laporanTugasAkhir-13521081-FINALFINAL.docx.pdf` — Predecessor thesis (Seto et al.)
- `13522108-ProposalTA-signed.pdf` — This thesis proposal

---

## 2. Environment Setup

Runs on **vast.ai** GPU instance (RTX 4090). On every new instance:

```bash
bash setup_runpod.sh
```

Installs Python 3.11 (deadsnakes PPA), creates `/workspace/venv311`, installs TF 2.15 + cuDNN 8.9 + PyTorch + HuggingFace, installs NeuralDecoder in editable mode.

**Always activate venv first:**
```bash
source /workspace/venv311/bin/activate
```

**Key compatibility notes:**
- TF 2.15 requires Python 3.8–3.11 (system Python 3.12 is incompatible)
- cuDNN: TF 2.15 needs `libcudnn.so.8`; PyTorch pulls cuDNN 9. Script pins `nvidia-cudnn-cu12==8.9.7.29` **after** PyTorch install.
- `lm_decoder` (C++ WFST binding) links against libtorch 1.13.1 — cannot coexist with torch 2.5.1 in the same process. Solved via subprocess separation (`eval_wfst_lm.py` → `rescore_nbest.py`).
- pybind11 in `LanguageModelDecoder/` upgraded from 2.9 → 3.0.3 (2.9 incompatible with Python 3.11).

---

## 3. Codebase Modifications

| File | Change |
|---|---|
| `NeuralDecoder/neuralDecoder/models.py` | Added `TransformerEncoder`, `ConformerEncoder` (with spatial attention) |
| `NeuralDecoder/neuralDecoder/neuralSequenceDecoder.py` | Conformer instantiation; cosine LR; early stopping; mixed precision |
| `NeuralDecoder/neuralDecoder/configs/model/conformer_stack_inputNet.yaml` | Conformer config |
| `NeuralDecoder/neuralDecoder/main.py` | Fixed 20GB GPU memory pool; mixed precision |
| `setup_runpod.sh` | Full env setup (Python 3.11 + TF 2.15 + PyTorch); cuDNN pin after PyTorch |
| `AnalysisExamples/eval_lm_pipeline.py` | Lexicon-constrained CTC beam search + N-best rescoring pipeline |
| `AnalysisExamples/eval_wfst_lm.py` | **NEW** — WFST pipeline: Conformer/GRU inference → lm_decoder beam search → N-best |
| `AnalysisExamples/rescore_nbest.py` | **NEW** — Subprocess rescorer: reads N-best JSON, scores with GPT-2/Gemma, outputs results |
| `LanguageModelDecoder/runtime/server/x86/pybind11/include/pybind11/` | Upgraded pybind11 2.9 → 3.0.3 for Python 3.11 compatibility |

---

## 4. Architecture Search Results (COMPLETE)

3-round Successive Halving over Transformer hyperparameters using 19 sessions.

**Round 3 winners:** `transformer_256d_4L_8H_512ff` (PER=0.498) and `transformer_512d_4L_8H_2048ff` (PER=0.501).

Results: `experiments/round3/final_results.json`

---

## 5. Full Training Results (COMPLETE)

| Model | Sessions | PER | Notes |
|---|---|---|---|
| **Conformer 512d+spatial** | **24** | **0.1654** | **Thesis contribution 1. Best model.** |
| GRU baseline (Willett et al.) | 24 | 0.1818 | Pre-trained checkpoint |
| Conformer 512d+LSO | 19 | 0.2130 | Without spatial attention |
| Transformer 512d+LSO | 19 | 0.2754 | Best pure Transformer |

**Key findings:** Cosine annealing > linear decay; LR=0.015 optimal; Conformer depthwise conv solved architectural bottleneck (+22.6% over best Transformer); spatial attention + 5 extra sessions pushed PER to 0.1654.

**Note:** Adam epsilon=0.1 inherited from Willett. Do NOT change without ablation.

Checkpoint: `experiments/24sess/conformer_spatial_24sess/ckpt-126000`

---

## 6. LM Pipeline History (Summarized)

### 6a. Attempt 1 — Post-hoc CMU dict DP (ABANDONED)
Custom Python CTC beam search followed by CMU dict exact-match DP. WER=0.958.
**Root cause:** PER=16.5% means a 6-phoneme word has only 35% chance of perfect phoneme sequence → exact match fails → syllable fragments emitted.

### 6b. Attempt 2 — Lexicon-Constrained Beam Search (PARTIAL SUCCESS)
Rewrote pipeline with prefix-trie lexicon constraint inside the beam. WER=0.587 (lexicon-only, beam=200).
**Bottleneck:** Oracle WER=0.481 — even perfect rescoring caps at 0.48 with current beam. Pure-Python approach without in-beam LM can't approach Willett's 0.118.

### 6c. Attempt 3 — WFST Pipeline (CURRENT BEST)
Switched to Willett's own `lm_decoder` C++ WFST binding. Faithful to his published pipeline.

---

## 7. WFST Pipeline Results (CURRENT — Session 9)

**Scripts:** `eval_wfst_lm.py` → `rescore_nbest.py`

### Conformer (PER=0.1654)

| Config | WER | CER | Oracle WER |
|---|---|---|---|
| WFST 3-gram only | 0.2219 | 0.1351 | 0.1177 |
| + GPT-2 124M (α=0.5) | 0.2197 | 0.1337 | — |
| + Gemma 3 270M (α=0.5) | 0.2870 | 0.1765 | — |

### GRU (PER=0.1818)

| Config | WER | CER | Oracle WER |
|---|---|---|---|
| WFST 3-gram only | 0.2204 | 0.1591 | **0.1033** |

### Key findings

- **GPT-2 gives negligible improvement** (0.2219 → 0.2197). Helped 139 utts, hurt 138 utts. Too weak to rerank reliably.
- **Gemma 3 270M hurts** (0.2219 → 0.2870). Overrides correct acoustic output.
- **GRU and Conformer tie at WER** despite Conformer having better PER. Conformer's peakier logits prune correct paths more aggressively; GRU's softer logits keep more diverse hypotheses alive (better oracle: 0.1033 vs 0.1177).
- **Bottleneck is 46.9% coverage failure** — correct answer not in the 100-best beam at all, so no rescorer can help. Root cause: 3-gram LM prunes correct paths because it assigns higher scores to plausible-but-wrong words.
- **18.6% reranking failure** — correct answer in beam but not at top-1. Would need a stronger LM to fix.
- Willett's 5-gram + OPT-6B achieves **WER=0.137** (confirmed in `AnalysisExamples/5gram+llm_rescoring.ipynb`).

---

## 8. Known Issues / Compatibility

- **TF checkpoint compat:** 24-sess checkpoint requires `_fix_all_checkpoint_compat` in `eval_lm_pipeline.py` (fixes Keras weight renaming between TF versions). Assigns 49–72 variables manually.
- **lm_decoder/torch ABI conflict:** `lm_decoder.so` links libtorch 1.13.1; importing torch 2.5.1 in same process causes `undefined symbol` crash. Fixed: `eval_wfst_lm.py` runs WFST decode, saves N-best to JSON, then calls `rescore_nbest.py` as a subprocess (no lm_decoder loaded there).
- **LD_LIBRARY_PATH timing:** Must be set before dynamic linker initializes. Both scripts use `os.execve` re-exec at startup.
- **Disk space:** Currently 71/128 GB used. Willett's 5-gram LM is 38 GB tar (~76 GB needed during extraction). **Upgrade to 256 GB before downloading.**

---

## 9. Next Steps (Priority Order)

### Step 1 — Upgrade disk + download 5-gram LM (HIGHEST IMPACT)
- Expand disk to 256 GB on vast.ai/RunPod
- Download Willett's `languageModel_5gram.tar.gz` (38 GB)
- Extract to `/workspace/speechBCI/languageModel_5gram/`
- Run: `python AnalysisExamples/eval_wfst_lm.py --lm-dir languageModel_5gram --lm none`
- Expected: WER drops from 0.222 → ~0.15–0.18 (attacks 46.9% coverage failure directly)
- Note: 5-gram dir must contain `G_no_prune.fst` for lattice rescoring; set `rescore=True` in `wfst_decode_all()`

### Step 2 — Tune acoustic_scale per model
- Run grid search: `--grid-search --acoustic-scales 0.2,0.3,0.4,0.5,0.7`
- Conformer may benefit from lower acoustic_scale (0.2–0.3) to dampen overconfidence
- GRU optimal is probably different from Conformer optimal

### Step 3 — Stronger rescoring LM
- Try **LLaMA 3 8B 4-bit** (needs `bitsandbytes`, HF token, ~5 GB with 4-bit)
- Add `--lm-id meta-llama/Meta-Llama-3-8B` to `rescore_nbest.py`
- Attacks 18.6% reranking failure; GPT-2/Gemma are too weak

### Step 4 — Conformer acoustic_scale tuning
- If Conformer still trails GRU after 5-gram, try temperature scaling (divide logits by T > 1 to soften before WFST)
- Or just use GRU for the LM pipeline since oracle is better

---

## 10. Useful Commands

```bash
# New pod: full environment setup
bash setup_runpod.sh && source /workspace/venv311/bin/activate

# WFST pipeline — Conformer, no rescoring
python AnalysisExamples/eval_wfst_lm.py --lm none

# WFST pipeline — Conformer + GPT-2 grid search
python AnalysisExamples/eval_wfst_lm.py --lm gpt2 --grid-search

# WFST pipeline — with 5-gram LM (after download)
python AnalysisExamples/eval_wfst_lm.py \
    --lm-dir /workspace/speechBCI/languageModel_5gram \
    --lm gpt2 --grid-search

# WFST pipeline — GRU baseline
python AnalysisExamples/eval_wfst_lm.py \
    --ckpt-dir /workspace/speechBCI/experiments/24sess/gru_1024u_5L_24sess \
    --output-dir /workspace/speechBCI/experiments/wfst_lm_gru \
    --lm none

# Rescore existing N-best with a new LM (skips inference + WFST, much faster)
python AnalysisExamples/rescore_nbest.py \
    --nbest-file experiments/wfst_lm/_nbest_tmp.json \
    --lm-id google/gemma-3-270m \
    --lm-tag gemma3_270m \
    --output-dir experiments/wfst_lm \
    --hf-token <HF_TOKEN>

# TensorBoard
tensorboard --logdir=/workspace/speechBCI/experiments --host=0.0.0.0 --port=6006
```

---

## 11. Summary of Best Results

### Phoneme Decoder (COMPLETE)

| Model | Sessions | PER | Notes |
|---|---|---|---|
| **Conformer 512d+spatial** | **24** | **0.1654** | **Thesis contribution 1** |
| GRU baseline (Willett et al.) | 24 | 0.1818 | Pre-trained checkpoint |
| Conformer 512d+LSO | 19 | 0.2130 | Without spatial attention |
| Transformer 512d+LSO | 19 | 0.2754 | Best pure Transformer |

### LM Pipeline (IN PROGRESS)

| System | WER | CER | Method |
|---|---|---|---|
| **WFST 3-gram (Conformer)** | **0.2219** | **0.1351** | **Current best** |
| WFST 3-gram + GPT-2 | 0.2197 | 0.1337 | Negligible gain |
| WFST 3-gram + Gemma 3 270M | 0.2870 | 0.1765 | Worse |
| WFST 3-gram (GRU) | 0.2204 | 0.1591 | Ties Conformer at WER |
| Lexicon beam (Python) | 0.587 | 0.300 | Previous attempt |
| CMU-dict DP | 0.958 | 0.489 | Original attempt |
| **Willett 5-gram + OPT-6B** | **0.137** | — | **Target** |
| Seto et al. (LLaMA 2) | ~0.170 | ~0.145 | Predecessor thesis |

---

## 12. Hardware

Current instance: RTX 4090 (24GB VRAM).

| Concern | Recommendation | Reason |
|---|---|---|
| **Disk** | **256 GB** | 5-gram LM = 38 GB tar; checkpoints 2–5 GB each; pip cache |
| **VRAM** | 24 GB sufficient | 4-bit LLaMA 3 8B fits (~5 GB); TF decoder ~4 GB |
| **RAM** | 64 GB | Large LM + TF data pipeline |
