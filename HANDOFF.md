# Speech BCI: Transformer Experiment Progress

**Last Updated:** 2026-04-05 (Session 8 — Lexicon-Constrained Beam Search IN PROGRESS)

Primary handoff for resuming thesis work. Covers goals, completed work, current state, and next steps.

---

## 1. Thesis Goal

Starting point: Willett et al. (2023) Speech BCI repo — GRU-based phoneme decoder for intracortical electrode recordings.

**Two thesis contributions:**

1. **Transformer Phoneme Decoder:** Replace GRU with Transformer encoder; find best architecture via Successive Halving. Goal: match/beat GRU PER.
2. **Full Speech Pipeline:** Integrate best decoder with a language model. Predecessor thesis (13521081) used n-gram + Transformer LM; this thesis improves on that.

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

This installs Python 3.11 (via deadsnakes PPA), creates `/workspace/venv311`, installs TF 2.15 + cuDNN 8.9 + PyTorch + HuggingFace, and installs NeuralDecoder in editable mode.

**Always activate venv first:**
```bash
source /workspace/venv311/bin/activate
```

**Key compatibility notes:**
- TF 2.15 requires Python 3.8–3.11 (system Python 3.12 is incompatible)
- `tensorflow[and-cuda]==2.15` bundles cuDNN 9 by default; script pins `nvidia-cudnn-cu12==8.9.*` (TF 2.15 needs `libcudnn.so.8`)
- GPU memory: `main.py` uses a fixed 20GB pool (not `memory_growth`) to prevent BFC allocator fragmentation during long training
- LM eval: `eval_lm_pipeline.py` sets `memory_growth=True` instead, to share GPU between TF decoder and PyTorch LM

---

## 3. Codebase Modifications

| File | Change |
|---|---|
| `NeuralDecoder/neuralDecoder/models.py` | Added `TransformerEncoder`, `ConformerEncoder` (with `ConformerConvModule`, `ConformerBlock`); spatial attention (`spatialAttention=True`); output Dense forced to float32 for CTC stability under mixed precision |
| `NeuralDecoder/neuralDecoder/neuralSequenceDecoder.py` | Added Transformer/Conformer instantiation; cosine annealing LR (`lrScheduleType: cosine`); early stopping (`earlyStopPatience`); `LossScaleOptimizer` wrapping; dtype-safe normalization |
| `NeuralDecoder/neuralDecoder/configs/model/conformer_stack_inputNet.yaml` | New config: Conformer (`modelType: conformer`, `convKernelSize: 31`) |
| `NeuralDecoder/neuralDecoder/configs/model/transformer_stack_inputNet.yaml` | New config: Transformer (includes `gradientCheckpointing: false` default) |
| `NeuralDecoder/neuralDecoder/configs/config.yaml` | Added `earlyStopPatience: 0`, `mixedPrecision: false`, `lrScheduleType: polynomial` defaults |
| `NeuralDecoder/neuralDecoder/main.py` | Fixed 20GB GPU memory pool; mixed precision activation |
| `setup_runpod.sh` | Full environment setup (Python 3.11 + TF 2.15 + PyTorch) |
| `AnalysisExamples/eval_lm_pipeline.py` | LM pipeline: CTC beam search → CMU dict DP → GPT-2/Gemma scoring |
| `AnalysisExamples/run_round{1,2,3}_experiments.py` | Successive Halving runners with OOM auto-retry |
| `AnalysisExamples/run_final_training.py` | Final training (top 2 configs, 100k batches) |
| `AnalysisExamples/run_cosine_ablation.py` | Cosine LR ablation (256d, 200k steps) |
| `AnalysisExamples/run_probe_experiments.py` | 30k-batch LR/dropout probe runs |
| `AnalysisExamples/run_512d_lr015_lso.py` | 512d + LSO (best Transformer result) |
| `AnalysisExamples/run_conformer_512d_lr015_lso.py` | Conformer 512d + LSO |
| `AnalysisExamples/eval_baseline.py` | GRU baseline evaluation |
| `AnalysisExamples/error_analysis.py` | Needleman-Wunsch alignment; per-phoneme/session error breakdown |

---

## 4. Architecture Search Results (COMPLETE)

3-round Successive Halving over Transformer hyperparameters (`d_model`, `num_layers`, `nhead`, `d_ff`) using 19 sessions.

| Round | Configs | Batches | Settings |
|---|---|---|---|
| 1 | 16 | 1,000 | bs=64, lr=0.001 |
| 2 | 8 | 5,000 | bs=32, lr=0.0005, grad ckpt |
| 3 | 4 | 20,000 | bs=32, lr=0.0005, grad ckpt, mixed precision |

**Round 3 winners:** `transformer_256d_4L_8H_512ff` (PER=0.498) and `transformer_512d_4L_8H_2048ff` (PER=0.501).

Results: `experiments/round3/final_results.json`

---

## 5. Full Training Results (COMPLETE)

All experiments used 19 sessions for fair comparison against GRU baseline, then extended to 24 sessions for the final conformer.

| Model | PER | Notes |
|---|---|---|
| **Conformer 512d+spatial, 24-sess** | **0.1654** | Best. Spatial attn. Config: `experiments/24sess/conformer_spatial_24sess/args.yaml` |
| GRU baseline (Willett et al.) | 0.1690 | Pre-trained checkpoint |
| Conformer 512d+LSO, 19-sess | 0.2130 | Conv kernel=31 |
| Transformer 512d+LSO | 0.2754 | Best pure Transformer |
| Transformer 256d (cosine, LR=0.015) | 0.3157 | Best 256d result |

**Key findings from training progression:**
- Cosine annealing outperforms linear decay (8.7% relative improvement)
- LR=0.015 outperforms LR=0.001 for both model sizes
- `LossScaleOptimizer` eliminated NaN instability under mixed precision (6% relative improvement)
- Scaling width/depth alone gave no gain — confirmed architectural bottleneck
- Conformer's depthwise conv solved the architectural bottleneck (22.6% relative improvement over best Transformer)
- Spatial attention + 5 extra sessions pushed Conformer PER from 0.2130 → 0.1654, beating the GRU

**Adam epsilon=0.1 note:** Inherited from Willett et al. All results use this. Do NOT change without a full ablation.

Checkpoint: `experiments/24sess/conformer_spatial_24sess/ckpt-126000`

---

## 6. LM Pipeline Results (COMPLETE, but suboptimal — see Section 8)

Integrated the 24-session Conformer with GPT-2 and Gemma 3 270M via N-best rescoring.

**Script:** `AnalysisExamples/eval_lm_pipeline.py`

| Component | PER | CER | WER | WPM |
|---|---|---|---|---|
| Conformer 24-sess (no LM) | 0.1654 | — | — | — |
| + GPT-2 124M | 0.1654 | 0.4886 | 0.9580 | 69.2 |
| + Gemma 3 270M | 0.1654 | 0.6141 | 1.1908 | 69.2 |

**Seto et al. results (predecessor thesis):** WER=~0.17, CER=~0.145, WPM=62.5 with LLaMA 2

Results: `experiments/lm_pipeline/lm_pipeline_results.json`

---

## 7. Known Issues

**VRAM OOM on long training:** Transformer attention is O(T²) memory; BFC allocator fragments over many steps. Mitigations: fixed 20GB pool (`main.py`), gradient checkpointing (`gradientCheckpointing=true`), mixed precision, OOM auto-retry in runner scripts.

**TF checkpoint compatibility:** The 24-session checkpoint was trained with TF 2.15. Loading it under a different TF version requires the `_fix_all_checkpoint_compat` function in `eval_lm_pipeline.py` (fixes Keras weight renaming: `kernel`→`_kernel` for EinsumDense, `layer_with_weights-N`→`_functional/_operations/N` for Sequential sublayers). Assigns 72 variables manually.

---

## 8. Discrepancies vs Seto et al. (Why Our WER Is So High)

Our WER (~0.96) is far worse than Seto's (~0.17). The root cause is a fundamental mismatch in how phonemes are converted to words.

| Aspect | Our Implementation | Seto et al. / Willett et al. |
|---|---|---|
| **Phoneme→word** | CMU dict DP post-hoc (after beam search) | Lexicon-constrained during beam search |
| **LM integration** | N-best rescoring (score completed sentences) | Shallow fusion (score at word boundaries *during* search) |
| **Word validity** | Hypotheses can contain non-words (syllable fragments) | All hypotheses are forced to be valid word sequences |
| **Search space** | Phoneme-level only; words emerge from DP lookup | Word-level search; phoneme-to-word transitions are explicit |
| **Lexicon type** | CMU dict (exact phoneme match required) | Lexicon FST (allows partial prefix matching during search) |
| **LM used** | GPT-2 124M / Gemma 3 270M | LLaMA 2 7B / GPT-2 (Seto); SRILM trigram 125k-word (Willett) |
| **Infrastructure** | Custom Python CTC beam + CMU dict | Kaldi-style lexicon decoder (SRILM in `LanguageModelDecoder/`) |

**Why the DP fails:** A 16.5% phoneme error rate means a 6-syllable word has only ~(0.835)^6 ≈ 35% chance of all phonemes being correct. The CMU dict requires an exact match — any single phoneme error causes the word to be unrecognized, and the DP falls back to outputting syllable fragments instead of words.

**Why Seto/Willett don't have this problem:** Their lexicon FST constrains the CTC beam so that every partial hypothesis is a valid word prefix. A phoneme error just selects a wrong-but-real word rather than producing a non-word fragment. The error is recoverable by the LM.

**The SRILM infrastructure is already in the repo** at `LanguageModelDecoder/` — Willett built it for trigram LM decoding. It was not used in our LM pipeline but is the correct foundation for a proper implementation.

---

## 8b. Session 8 — Lexicon-Constrained Beam Search (IN PROGRESS)

**Objective:** Rewrite the LM pipeline to be faithful to Seto + Willett. Root cause of WER=0.96 is post-hoc CMU-dict DP (phoneme errors break exact-match lookup and emit syllable fragments). Fix: make the beam search itself lexicon-constrained so every partial hypothesis is a valid word prefix.

**Plan file:** `/root/.claude/plans/snazzy-wishing-garden.md` (approved).
Design: pure-Python CMU-dict prefix trie + word-level CTC beam search, then N-best rescoring with GPT-2 124M / Gemma 3 270M, combined per Seto eq. III-3: `score = acoustic + α·log P_LM + β·word_insertion_bonus`.

### What was implemented
Modified `AnalysisExamples/eval_lm_pipeline.py`:
- **`LexiconTrie`** class: CMU dict → prefix trie keyed on stress-stripped phoneme tuples. Multi-pronunciation support. Full ~125k vocab.
- **Short-word junk filter** (`_SHORT_WORD_WHITELIST`): drops 1005 CMU 1–2-phoneme interjections (mm, oh, tew, reh, etc.); keeps real short words (a, i, be, by, do, go, etc. + contractions).
- **`lexicon_constrained_beam_search`** (replaces `ctc_prefix_beam_search` + `phones_to_words_dp`):
  - Beam key: `(committed_words, trie_node_id, last_emitted_idx)`
  - Value: `(log_prob_acoustic, n_words)`
  - Four transitions per frame: blank / repeat last nonblank / SIL-commits-word / phoneme-extends-trie (with implicit word boundary at word-end nodes).
  - Viterbi max-path (not the log-sum-exp pb/pnb split — acceptable simplification, verified on synthetic logits).
- **`score_nbest_with_lm`** + **`pick_best_hyp`** for Seto III-3 combine.
- New CLI: `--alpha`, `--beta`, `--beam-beta`, `--grid-search`, `--alphas`, `--betas`, `--max-utts`, `--lm none`.
- Oracle best-of-N-best WER debug output.

**Constants (confirmed correct):** `BLANK_IDX=40, SIL_IDX=39, PHONE_CLASS_OFFSET=0`. TFRecord `seqClassIDs` are 1-indexed (1–40), then `-1` is applied to get internal 0–39 matching `PHONE_DEF_SIL`. `neuralSequenceDecoder.py:892` confirms `blank_index=-1` in ctc_loss_v2 (last class = blank). Verified empirically: class 40 is the most frequent argmax on real logits (blank dominates, as expected in CTC).

### Rabbit holes (for the record)
1. **Wrong off-by-one hypothesis.** Seeing seqClassIDs 1–40 in TFRecords, I initially guessed `BLANK_IDX=0`. WER got *worse* (1.05 → 2.26). Reverted after checking argmax class frequency. Fix: original `BLANK_IDX=40` is correct.
2. **cuDNN version mismatch.** venv311 had libcudnn.so.9, TF 2.15 needs libcudnn.so.8. GPU silently fell back to CPU (25-min hang). Fix: `pip install 'nvidia-cudnn-cu12==8.9.7.29' --force-reinstall --no-deps`.
3. **Over-segmentation WER>1.0.** Beam emitted short junk like "mm"/"tew". Fix: short-word whitelist filter (above).

### Current results (preliminary, full 880-utt test)

| Config | CER | WER | Notes |
|---|---|---|---|
| Old: N-best rescoring + CMU-DP (GPT-2) | 0.489 | **0.958** | Session 7 baseline |
| **Lexicon-only, no LM, beam=200, β=0** | **0.300** | **0.587** | **This session, 322s on GPU** |
| Oracle (best-of-N-best, avg 87.5 hyps/utt) | 0.260 | 0.481 | Ceiling with current beam |
| Seto et al. target | ~0.145 | ~0.170 | LLaMA 2 + lexicon FST |

**Lexicon constraint alone dropped WER 0.96 → 0.59 (–39% relative).** Matches the plan's prediction (~0.40–0.60 without LM).

### Known bug (blocker for next run)
`eval_lm_pipeline.py:875` — `UnboundLocalError: cannot access local variable 'results'` when oracle block writes to `results` before `results = {...}` is initialized. Move oracle assignment *after* the main `results = {...}` dict is built, or initialize `results = {}` earlier. Pure reporting bug — the decoder completed successfully before it crashed.

### Clues / diagnosis so far
- **PER ≠ uniform across utterances.** First 50 test utts have PER=0.276 (hard subset) vs full-set internal PER=0.1654. Hard lexicon constraint struggles most where PER is high: correct phoneme path falls out of the beam entirely.
- **Oracle WER=0.481 with avg 87.5 hyps/utt** means even with a perfect LM, current beam caps us at ~0.48. To reach Seto's 0.17 we likely need either (a) a larger beam (500–1000) to keep the correct path alive longer, or (b) log-sum-exp pb/pnb prefix beam (standard CTC) instead of Viterbi max-path, which merges alignment variants and improves recall of correct paths.
- **LM headroom:** gap between top-1 (0.587) and oracle (0.481) is 0.106 WER. A good LM rescorer should close most of that; remaining gap to Seto (0.17) must come from enlarging the beam / tightening the acoustic model.

### Next steps (in order)
1. **Fix the `results` UnboundLocalError** at line 875 (one-line fix).
2. **Re-run full test + GPT-2 + Gemma rescoring + α/β grid search.** Command:
   ```bash
   python AnalysisExamples/eval_lm_pipeline.py --lm both --beam-size 200 \
       --grid-search --alphas 0.3 0.5 0.8 1.2 --betas 0.0 0.5 1.0 2.0 \
       --hf-token $HF_TOKEN
   ```
   Expected: WER ~0.30–0.40 after rescoring.
3. **If WER still >0.35 after LM:** bump beam to 500, consider switching to log-sum-exp pb/pnb prefix beam (bigger change, matches standard CTC prefix beam search).
4. **Stretch:** integrate KenLM 3-gram *inside* the beam (Willett's approach). Faithfulness to Willett improves further; beam prunes to LM-likely continuations earlier.

---

## 8c. Diagnosis — Is the Bottleneck the Decoder or the LM Pipeline?

**It's the LM/beam, not the phoneme decoder.** Arithmetic:

| System | PER | WER |
|---|---|---|
| Our Conformer | 0.165 | 0.587 (lexicon-only) / 0.481 (oracle) |
| Willett GRU (offline, his paper) | 0.169 | **0.118** |
| Seto (pretrained GRU + LLaMA 2) | ≈0.17 | **≈0.170** |

Same PER, ~3–5× worse WER. The phoneme decoder is doing its job. A cheap sanity check: plug the Willett GRU checkpoint into our pipeline — if we get WER ≈ 0.55–0.60, the decoder is absolved; if Willett's GRU gives WER 0.12 through our pipeline, then something is wrong with the Conformer logits.

### What's different — ours vs Willett's offline decoder

| Aspect | Ours | Willett |
|---|---|---|
| Decoder infra | pure-Python lexicon trie | Kaldi HCLG WFST (H+C+L+G composed) |
| LM inside beam | none | SRILM **3-gram, 125k vocab**, shallow-fused per step |
| Beam width | 200 hyps, avg 87 survive | Kaldi `beam=17` → lattices in the thousands |
| Prefix beam math | Viterbi max-path | **log-sum-exp over (p_blank, p_nonblank)** |
| Acoustic scale | unscaled frame log-probs summed | tuned `acoustic-scale` (0.1–1.0) |
| Word-insertion penalty | β, untuned | tuned on validation |

### Ours vs modern "best practice"

| Aspect | Ours | Best practice |
|---|---|---|
| Library | hand-rolled | `pyctcdecode` or `flashlight-text` (both handle KenLM + lexicon out of the box) |
| In-beam LM | none | KenLM 4- or 5-gram, per-step |
| Prefix merging | Viterbi | log-sum-exp (p_b, p_nb) |
| Beam width | 200 | 500–1500 |
| Rescoring LM | GPT-2 124M / Gemma 270M | LLaMA 3 8B 4-bit or domain-finetuned |

### Recommended surgery (in order of expected WER impact)

1. **Log-sum-exp prefix beam** replacing Viterbi max-path (largest correctness fix — Viterbi undercounts long words).
2. **Beam 500** and measure oracle WER. If oracle drops, keep going; if flat, the in-beam LM is the missing ingredient.
3. **KenLM 3-gram inside beam** (Willett-faithful; `pyctcdecode` reads `.arpa` directly).
4. **Only then** sweep α/β for large-LM rescoring.
5. Swap GPT-2 for **LLaMA 3 8B 4-bit** to close the Seto-style gap.

Steps 1–3 should take us from WER 0.59 → ~0.25. LLaMA-scale rescoring is the last 0.05–0.08.

---

## 8d. Is Seto's Thesis Trustworthy?

Honest assessment of Seto et al.'s methodology.

### Against fabrication (probable)
- Their WER (0.17) is **worse than Willett's** (0.118). Fabricators invent improvements, not regressions.
- Willett's **Kaldi HCLG + SRILM pipeline is literally in this repo** (`LanguageModelDecoder/`). Seto almost certainly used it as the backend — no need to re-implement anything to reach 0.15–0.20 WER.
- Wrong hyperparameter descriptions (e.g. LLaMA 7B config) are sloppiness, not dishonesty — standard for Indonesian BSc theses.
- WER 0.17 on this dataset is reachable with pretrained GRU (PER≈0.17) + Willett's Kaldi beam + LLaMA 2 7B as a rescorer.

### Toward "their 'shallow fusion' is really N-best rescoring"
- "Shallow fusion" vs "N-best rescoring" is routinely conflated by students — both apply eq. III-3, just at different granularities.
- **LLaMA 2 7B is extremely awkward to shallow-fuse inside a beam.** Per-step token scoring with a 7B model on 500+ beam entries is ≥100× slower than the acoustic decoder and usually not worth it. *Everyone* who uses a big LLM with CTC does it as rescoring in practice.
- Wrong LLaMA config details correlate with "wrote 'shallow fusion' because the references used it" rather than "implemented real per-step log-prob fusion with a 7B model."

### Most probable story
Seto ran Willett's existing Kaldi HCLG beam (with built-in SRILM 3-gram) to produce an N-best word list, re-ranked the N-best with LLaMA 2 7B log-probabilities, combined via eq. III-3, and called the whole thing "shallow fusion" in the thesis. That architecture yields WER ≈ 0.17 on this dataset, matches the existing repo infrastructure, and matches a bachelor-level engineering budget.

### Implication for us
**We don't need to match their word "shallow fusion" literally.** We need (a) Willett's Kaldi HCLG beam or our Python equivalent with a KenLM n-gram inside the beam, plus (b) LLaMA-or-similar N-best rescoring on top. That's what they probably did, regardless of terminology.

### How to verify (cheap)
1. Inspect `LanguageModelDecoder/` — if it has per-step LLaMA hooks, it's real shallow fusion; if only a rescoring entrypoint, it's rescoring.
2. Check if Seto has a GitHub companion repo.
3. Replicate: stand up Willett's Kaldi beam + LLaMA 2 rescoring — if we get WER ≈ 0.17, the number is real regardless of terminology.

---

## 9. TODO: Proper LM Pipeline (Thesis Contribution 2)

To match Seto's WER (~0.17), the pipeline needs lexicon-constrained word-level beam search with shallow fusion. Priority order:

### Step 1 — Build lexicon FST from CMU dict
- Convert CMU pronunciation dictionary to a phoneme-sequence → word mapping
- Represent as a trie or WFST: given partial phoneme sequence, return valid word completions
- Libraries: `openfst`, `pynini`, or a pure-Python trie (simpler, sufficient for prototype)

### Step 2 — Implement lexicon-constrained CTC beam search
- Modify the CTC beam search to track word-level state alongside phoneme state
- At each step: only extend beams with phonemes that are valid prefixes of some word in the lexicon
- At word boundaries (end of a word phoneme sequence): record the completed word and reset word-level state
- This is the core change — replaces the CMU dict DP entirely

### Step 3 — Integrate LM as shallow fusion at word boundaries
- When a word boundary is reached in the beam, query the LM: `P(next_word | previous_words)`
- Combine: `score = alpha * acoustic_score + beta * lm_score + gamma * word_bonus`
- The `word_bonus` penalizes fragmented phoneme sequences (encourages completing words)
- Tune alpha/beta/gamma on the validation set

### Step 4 — LM selection
- GPT-2 or LLaMA 3 8B for shallow fusion (causal, efficient prefix scoring)
- LLaMA 3 requires HuggingFace token and ~16GB VRAM; use 4-bit quantization (bitsandbytes) to fit on RTX 4090 alongside TF decoder
- Alternatively: use the existing SRILM trigram LM in `LanguageModelDecoder/` as a first test (no GPU needed, very fast)

### Step 5 — Evaluate and compare
- Report CER, WER, WPM vs our current numbers and vs Seto's results
- Ablate: trigram LM vs GPT-2 vs LLaMA to isolate LM contribution from decoder improvement

**Expected outcome:** Step 1+2 alone (lexicon-constrained search, no LM) should drop WER dramatically (from ~0.96 to ~0.40–0.60) just by eliminating non-word fragments. Adding shallow fusion (Step 3+4) should bring it to ~0.20–0.30 range.

---

## 10. Useful Commands

```bash
# New pod: full environment setup
bash setup_runpod.sh

# Activate venv (required before any script)
source /workspace/venv311/bin/activate

# LM pipeline evaluation (current N-best rescoring version)
python AnalysisExamples/eval_lm_pipeline.py \
    --lm both --beam-size 50 --lm-nbest 30 \
    --hf-token <HF_TOKEN>

# TensorBoard
tensorboard --logdir=/workspace/speechBCI/experiments --host=0.0.0.0 --port=6006

# GRU baseline evaluation
python AnalysisExamples/eval_baseline.py

# Error analysis (Transformer vs GRU)
python AnalysisExamples/error_analysis.py
```

---

## 11. Summary of Best Results

### Phoneme Decoder

| Model | Sessions | PER | Notes |
|---|---|---|---|
| **Conformer 512d+spatial** | **24** | **0.1654** | **Thesis contribution 1 — beats GRU** |
| GRU baseline (Willett et al.) | 19 | 0.1690 | Pre-trained checkpoint |
| Conformer 512d+LSO | 19 | 0.2130 | Without spatial attention |
| Transformer 512d+LSO | 19 | 0.2754 | Best pure Transformer |

**Thesis contribution 1: COMPLETE** — Conformer 24-sess (PER=0.1654) beats GRU (PER=0.1690).

### LM Pipeline

| LM | CER | WER | WPM | Method |
|---|---|---|---|---|
| GPT-2 124M (ours) | 0.4886 | 0.9580 | 69.2 | N-best rescoring (post-hoc DP) |
| Gemma 3 270M (ours) | 0.6141 | 1.1908 | 69.2 | N-best rescoring (post-hoc DP) |
| LLaMA 2 (Seto et al.) | ~0.145 | ~0.170 | 62.5 | Shallow fusion (lexicon-constrained) |

**Thesis contribution 2: IN PROGRESS** — Pipeline works end-to-end but WER is high due to post-hoc phoneme→word conversion. Proper lexicon-constrained shallow fusion (Section 9) is the next step.

---

## 12. Hardware Recommendations

Current instance: RTX 4090 (24GB VRAM), ~100GB disk.

| Concern | Recommendation | Reason |
|---|---|---|
| **Disk** | **300–500 GB** | LLaMA 3 8B ≈ 16GB, LLaMA 3 70B ≈ 140GB, checkpoints accumulate at 2–5GB each, pip cache fills fast (5–8GB). The 100GB disk ran out during TF+PyTorch install. |
| **VRAM** | **40–80 GB (A100/H100)** | Running TF decoder (needs ~4GB) + LLaMA 7B in float16 (14GB) + KV cache during beam search fills 24GB. 4-bit quant can fit on 24GB but limits batch size and speed. |
| **RAM** | **64 GB** | Loading large LM tokenizer + model into CPU RAM before GPU transfer; also needed for CMU dict (large phoneme trie) + TF data pipeline |
| **CPU cores** | 8+ | CTC beam search is single-threaded Python; multiple cores help if batching multiple utterances in parallel |

**Short-term (RTX 4090 is sufficient):** Use 4-bit quantized LLaMA 3 8B via `bitsandbytes` to fit on 24GB. Upgrade disk to at least 300GB.

**Long-term (for proper benchmarking):** A100 40GB allows running LLaMA 7B in float16 + TF decoder simultaneously with no quantization compromise.
