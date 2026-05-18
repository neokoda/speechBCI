# TRACKER.md

Gap checklist. Move items from `[ ]` (pending) → `[x]` (done) as they close. Keep this file under 200 lines.

**Updated:** 2026-05-18 (Session 19 start). Linked: [EXPERIMENTS.md](./EXPERIMENTS.md), [HANDOFF.md](./HANDOFF.md).

---

## Priority A — E2E push (whisper + cohere, this session's focus)

### A1 — Session-sliced eval support in `AnalysisExamples/e2e/eval.py`
- [ ] Track `session_idx` per utterance in `details`
- [ ] Compute corpus-level WER/CER for `willett_4_18` (sessions 4–18), `willett_19`, `all_24`
- [ ] Emit slices as JSON keys `wer_<slice>`, `cer_<slice>`, `n_<slice>`

### A2 — Full-set eval on existing best E2E checkpoints
- [x] Eval `experiments/e2e_v7/best` → `experiments/e2e_v7/eval_full.json` (24 sess + slices)  — WER@all_24=0.2053, willett_19=0.2062, willett_4_18=0.1716
- [x] Eval `experiments/e2e_v6/best` → `experiments/e2e_v6/eval_full.json`  — WER@all_24=0.2157
- [ ] Eval `experiments/e2e_canary_ctc/best` → `experiments/e2e_canary_ctc/eval_full.json`
- [ ] Eval `experiments/e2e_granite/best` → `experiments/e2e_granite/eval_full.json`

### A3 — Maximally push Whisper (v7 → v8)
v7 trajectory: WER still tightening at the last eval (0.2055 at step 14500 / 15000). LR profile peaked at: encoder=6.9e-5, projector/cross-attn=1.0e-3, lora=1.75e-4 with cosine decay over 15k steps.
- [ ] v8a — extend from v7/best for +15k steps with **higher base/end LR** (mimic two-stage finding: start 4e-2 → end 4e-3 over decay): encoder 2e-4 → 5e-5, projector 1.5e-3 → 3e-4, lora 3e-4 → 5e-5. Lower if loss diverges.
- [ ] v8b — if v8a stalls, try fresh 25k-step run with same LRs but stronger reg (label_smoothing=0.1 → 0.15, lora_dropout 0.1 → 0.2)
- [ ] v8c — if v8b stalls, try data augmentation pass (more SpecAugment masks, white_noise_sd sweep)
- **Stop condition:** if no run beats v7's 0.2055 by step 7000 of its budget, stop pushing Whisper and accept v7 as headline.

### A4 — Cohere transcribe-03-2026
- [ ] Verify HuggingFace availability: `CohereLabs/cohere-transcribe-03-2026`
- [ ] Inspect architecture (config.json) — confirm enc-dec audio model with cross-attention decoder
- [ ] Write `AnalysisExamples/e2e/cohere_model.py` mirroring `whisper_model.py`
- [ ] Train with the best Whisper config from A3
- [ ] Full-set eval + 3-schema slices

---

## Priority B — Two-stage gap closure

### B1 — 19-session models through 5-gram WFST
- [ ] GRU 19sess + 5-gram
- [ ] Conformer vanilla 19sess + 5-gram
- [ ] Conformer +spatial 19sess + 5-gram

### B2 — 3-schema reporting for two-stage
- [ ] Add session slicing to `AnalysisExamples/eval_wfst_lm.py` (mirrors A1 for the two-stage path)
- [ ] Re-emit results for all 24-sess two-stage runs

### B3 — Remaining LM rescore combos
- [ ] Fine-tuned LLaMA-2 7B rescoring on **GRU 24sess** (currently only Conformer-spatial)
- [ ] Write up beam-24/temp-1.5 sweep from `experiments/wfst_5gram_*_beam24_nb200/`, `*_temp1p5/` into a result row

### B4 — (Blocked) Full lattice rescoring
- [ ] Schedule a 128 GB-RAM instance
- [ ] Lattice rescoring with G_no_prune.fst

---

## Priority C — Speed / supporting (proposal §540–598)

- [ ] C1 — Build `AnalysisExamples/measure_speed.py`
- [ ] C2 — Measure RTF + WPM for: GRU+5gram, Conformer-spatial+5gram, Conformer-spatial+5gram+ftLLaMA, E2E Qwen v5, E2E Canary, E2E Granite, E2E Whisper v7 (and v8 if it wins)
- [ ] C3 — (Optional) int8 / int4 quantization sweep on the E2E models

---

## Priority D — Analysis chapter

Pre-req: A2 done.

- [ ] D1 — Pick winning audio-FM after A2 (likely Whisper v7 unless v8 or Cohere wins)
- [ ] D2 — `AnalysisExamples/analysis/probe_layers.py` — linear probe per layer for phoneme + word identity, on winning audio-FM and on Qwen 3.5-0.8B (E2E-2)
- [ ] D3 — `AnalysisExamples/analysis/visualize_cross_attn.py` — per-head attention entropy + average attention matrix
- [ ] D4 — Side-by-side LibriSpeech vs ECoG attention pattern (qualitative figure)

---

## Priority E — Docs

- [ ] E1 — Update EXPERIMENTS.md after every completed checkbox above
- [ ] E2 — Append Session 19 entry to HANDOFF.md once a result lands
