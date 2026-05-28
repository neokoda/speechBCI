# HANDOFF — Speed & storage benchmark (Session 21, paused 2026-05-28)

This is a **task-specific handoff**, separate from the main `HANDOFF.md`. It exists because Session 21 was started on a pod that turned out to be unusable (vLLM holding the GPU, broken venv, unbuilt WFST decoder) and the user is moving to a fresh machine. All decisions, blockers, and remaining work are captured below so the next session can resume directly.

---

## 1. What we are trying to do

Populate **EXPERIMENTS.md §4** (currently TBD) with **storage size, WPM, and RTF** for every published model so the thesis can argue accuracy↔real-time trade-offs.

Models to benchmark:

- **Two-stage** (5 rows): LM1 (GRU + 5-gram), LM2 (Conformer-spatial + 5-gram), LM3 (Conformer-vanilla + 5-gram), LM-x (GRU + 5-gram + LLaMA-2-7B base rescore), LM6 (Conformer-spatial + 5-gram + LLaMA-2-7B base rescore).
- **E2E** (6 rows): `e2e_v5` (Qwen-LLaVA), `e2e_v6` (Whisper-medium.en), `e2e_v7` (Whisper-large-v3 — headline), `e2e_cohere_v3_ext3` (Cohere), `e2e_canary_ctc` (Canary), `e2e_granite` (Granite).

---

## 2. Decisions locked in with the user (do NOT re-litigate)

| Decision | Value | Rationale |
|---|---|---|
| **`T_audio` for RTF** | `num_neural_bins × 20 ms` per utterance | Matches Seto thesis §III.2 and Metzger 2023. Willett's `tx1..tx4`/`spikePow` are **non-overlapping** 20 ms windows (confirmed Seto §II.3). Conformer's internal 4× subsampling is downstream and doesn't change `T_audio`. There is no audio in BCI — trial duration IS the denominator. |
| **WPM denominator** | Same `T_audio` (trial-minutes) | Same as Seto. Note `eval_wfst_lm.py:115-120` already computes WPM this way using ground-truth word count; we keep that. |
| **Eval subsets** | Two: (a) first 100 utts of all_24 test, fixed deterministic; (b) full `willett_4_18` (600 utts) | First100 matches Seto + the existing §4 stub. willett_4_18 matches a slice column already in §3. |
| **Storage = full deployable size** | Includes the FM backbone for E2E (not just the trainable checkpoint) and includes TLG.fst + words.txt + LLaMA-2 for two-stage. Every file required at inference. | What you'd ship. Reported as a sub-table with per-component lines and a total. |
| **Hardware** | Single GPU workstation. TF + LLaMA rescore + E2E generate on GPU; WFST on CPU (lm_decoder is C++/libtorch-1.13, CPU-only by design). | Apples-to-apples wall-clock per utterance on one machine. Mirrors how every published clinical speech BCI (Willett 2023, Metzger 2023, UC Davis 2025) actually runs. Per web search there are **no portable speech BCI devices shipping mid-2026** — Neuralink/Synchron/Paradromics still stream off-body — so embedded HW constraints don't apply yet; RTF < 1 (Seto's framing) is the relevant bar. |

---

## 3. Files to create (next session)

### `AnalysisExamples/measure_speed.py` (new)

One script, three pipeline modes (`--pipeline {wfst, wfst+rescore, e2e}`), one JSON output per (model, subset). Must **reuse** existing code paths so timing has no implementation drift from the published eval:

- **Two-stage path:** import + call the same functions `eval_wfst_lm.py` uses (`lmDecoderUtils.build_lm_decoder` → TF forward → `lmDecoderUtils.lm_decode` → optional `rescore_with_slices.py` subprocess). Instrument three regions with `time.perf_counter()`:
  - `t_neural` — TF forward (GPU)
  - `t_wfst` — `lm_decode` call (CPU)
  - `t_rescore` — LLaMA-2 7B forward over the n-best (GPU). Only LM-x and LM6.
- **E2E path:** import + call the same model factory as `e2e/eval.py` (whisper/cohere/canary/granite/llava branches all exist), `batch_size=1`, `beam=1`. Time the `model.generate` call.
- **Common:** 5-utt warmup, then per-utt `time.perf_counter()`. Record `t_total`, `n_words_ref`, `n_words_hyp`, `n_bins`. Compute:
  - `T_audio_i = n_bins_i × 0.02`
  - `RTF_i = t_total_i / T_audio_i`
  - `WPM_i = n_words_hyp_i / (T_audio_i / 60)`
  - Report per-utt mean + median **and** corpus aggregate (`Σt / ΣT_audio` and `Σwords / (ΣT_audio/60)`).
- **Subset selector:** `--subset {first100, willett_4_18}`. willett_4_18 reuses `WILLETT_4_18` constant added to `e2e/eval.py` in S19.
- **Output:** `experiments/<run>/speed_<subset>.json` with `{rtf_corpus, wpm_corpus, rtf_mean, wpm_mean, n, breakdown: {t_neural, t_wfst, t_rescore}, hardware: {gpu, cpu_model}}`.

### `AnalysisExamples/measure_storage.py` (new, tiny)

Walk each model's required files, `os.path.getsize`-sum.

- Two-stage: phoneme decoder ckpt dir + `speech_5gram/lang_test/{TLG.fst, words.txt, units.txt}` + (LM-x/LM6) LLaMA-2 7B weights.
- E2E: `experiments/<run>/best/checkpoint.pt` + HF backbone snapshot dir.
- Output: rows appended into `EXPERIMENTS.md §4` and a sidecar JSON.

### `EXPERIMENTS.md §4` update

Replace the TBD placeholder with **two tables** (storage, then speed) and protocol footnotes:
- Storage table cols: `Model | Component | MB | Total`.
- Speed table cols: `Model | Subset | RTF (corpus) | WPM (corpus) | t_neural | t_wfst | t_rescore | Hardware`. Two rows per model (first100 + willett_4_18). Hardware string autofilled by script.

---

## 4. Blockers that killed Session 21 on the old pod

The pod we were on (RunPod L40S, 200 GB volume) failed five preflight checks. The new machine needs ALL of these fixed before any timing runs.

| # | Blocker | Verification command | Fix |
|---|---|---|---|
| 1 | **`/workspace/venv/bin/python` missing** | `ls /workspace/venv/bin/python` | Recreate per S20: `python3.12 -m venv /workspace/venv && /workspace/venv/bin/pip install torch==2.4.1+cu124 transformers==5.6.2 peft==0.19.1 tensorflow==2.15.* numpy<2 scipy<1.13 jiwer sentencepiece accelerate librosa soundfile`. ~10 min. |
| 2 | **`lm_decoder` C++ ext not built** — only sources at `LanguageModelDecoder/runtime/server/x86/python/lm_decoder.cc`, no `.so` anywhere | `find /workspace/speechBCI -name 'lm_decoder*.so'` (should return a path) | Follow `LanguageModelDecoder/README.md` Step 3: cmake build with pybind11. Links libtorch 1.13.1 — version-sensitive. Budget 20–30 min and watch for cmake errors. **Required for ALL two-stage timing (LM1–LM6).** |
| 3 | **GPU was occupied by `VLLM::EngineCore` PID 295 holding 41.5 GB / 46 GB** | `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` (should show no procs, or fit your model) | Should be resolved by moving machines. Verify on the new machine before pulling models. |
| 4 | **`HF_TOKEN` not set** — Cohere and LLaMA-2 are gated | `[ -n "$HF_TOKEN" ] && echo ok` | `export HF_TOKEN=hf_...` (user has the token). Persist to shell rc or to `/workspace/.bashrc` so re-shells don't re-block. |
| 5 | **Canary eval is broken** (S20 dilemma 1 — runaway generation, full-set WER ≈ 1.8 with `transformers 5.6.2`) | Run `eval.py --model-type canary` on `e2e_canary_ctc/best`; if WER > 1, broken. | Two options: (a) accept and footnote — RTF is still valid (it's wall-clock), WPM-from-hyp is misleading because hyp word count is inflated; or (b) patch `canary_model.generate()` to use `tokenizer.apply_chat_template(..., enable_thinking=False)`. Pick (a) if pressed for time. |

---

## 5. Pre-flight pulls on the new machine

Once the five blockers are clear, pull only what's needed, in this order, to stay under quota.

### Storage budget (verified by scanning Dryad API + gdrive backup on 2026-05-28)

| Asset | Source | Size | For |
|---|---|---|---|
| `e2e_cohere_v3_ext3/best/checkpoint.pt` | `gdrive:speechBCI_backup/experiments/e2e_cohere_v3_ext3/` | **390 MB** | Cohere timing — local copy on old pod was deleted (only 1.2 MB skeleton present) |
| `languageModel_5gram.tar.gz` | Dryad version 253293 | **38.2 GB tar** (extracts to TLG.fst + G.fst + G_no_prune.fst + words.txt; extracted total ≈ 80 GB — but we only need TLG.fst + words.txt for `--lm none` decoding) | LM1, LM2, LM3, LM-x, LM6 timing |
| LLaMA-2 7B base safetensors | HF `meta-llama/Llama-2-7b-hf` | ~13 GB | LM-x, LM6 |
| `openai/whisper-medium.en` | HF | ~1.5 GB | v6 |
| `openai/whisper-large-v3` | HF | ~3 GB | v7 |
| `Qwen/Qwen3.5-0.8B-Base` | HF | ~1.6 GB | v5 |
| `CohereLabs/cohere-transcribe-03-2026` (gated) | HF | ~4 GB | Cohere |
| `nvidia/canary-qwen-2.5b` | HF | ~5 GB | Canary |
| `ibm-granite/granite-speech-4.1-2b` | HF | ~5 GB | Granite |
| **Total simultaneous** | | **~71 GB** | |

If the new machine has < ~80 GB free, do **pull → time → delete** in three phases:
1. Pull 5-gram tar; extract only `TLG.fst` + `words.txt` + `units.txt`; delete tar; time WFST (LM1/2/3); keep TLG for next phase.
2. Pull LLaMA-2 7B; time rescore (LM-x, LM6); delete LLaMA + TLG.
3. Pull E2E FMs in batches; time E2E rows; delete each FM as you finish its row.

### Dryad download

The README at `/workspace/speechBCI/README.md` line 8 mentions `languageModel_5gram.tar.gz` exists. Direct API confirmation done in S21:

```bash
# Version with 5gram tar:
curl -s "https://datadryad.org/api/v2/versions/253293/files?per_page=100"
# languageModel_5gram.tar.gz = 38.246 GB (the file we need)
```

Download URL pattern (Dryad): use the `_links.stash:download` from the file metadata; or browse to `https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq` and pull from the web UI. `aria2c -x16 -s16` recommended (S20 confirmed HF/CDN downloads stall under default tools).

### gdrive backup state (verified 2026-05-28)

```
gdrive:speechBCI_backup/
├── data/
├── docs/
├── experiments/
│   ├── e2e_cohere_v3_ext3/  ← contains best/checkpoint.pt (390 MB), train.log, eval_full.json
│   ├── e2e_v7/, e2e_v6/, e2e_v5/, e2e_v4/, e2e_canary_ctc/, e2e_v8/, ...
│   ├── ctc_4l/
│   ├── bssf_5gram_llama2_*_redo/ (S20 LLaMA rescoring)
│   ├── bssf_ft_llama2_ckpt7000/  (LM7 — adapter NOT in this dir, only result.json)
│   └── ...
└── speech_5gram/
    ├── lang_test/words.txt        (1.8 MB)
    └── lang_test/G.fst            (5.1 GB)   ← G for rescoring, NOT TLG. Pull from Dryad instead.
```

**Key gap:** TLG.fst is NOT in the gdrive backup. Was never uploaded. Must come from Dryad.

---

## 6. State of the old pod (informational)

`/workspace` quota: 200 GB. At pause: ~110 GB free (after user freed ~58 GB; original was 32.5 GB free).

Local files of interest:
- `experiments/24sess/{gru_1024u_5L_24sess, conformer_vanilla_24sess, conformer_spatial_24sess}/` — phoneme decoder checkpoints; these are needed by the two-stage timing. **Verify these come across to the new machine.** Total ~17 MB so trivial.
- `experiments/e2e_v{4,5,6,7}/best/checkpoint.pt`, `experiments/e2e_canary_ctc/best/checkpoint.pt`, `experiments/e2e_granite/best/checkpoint.pt` — all needed.
- `experiments/wfst_*_24sess_*/wfst_results.json` — **already contain WPM** computed with the same formula. Useful as cross-check (the live-timed WPM should match these to within ±2%).

Stale dirs identified but not yet deleted: `experiments/e2e_0.8b_fixed` (3.8 GB), `experiments/*_smoke` (~2.8 GB), `experiments/ctc_encoder` (657 MB, possibly older version of `ctc_4l`). Worth verifying before deleting on the new machine.

---

## 7. Verification steps (final)

After implementation, before publishing:
1. Smoke-run `measure_speed.py --pipeline e2e --ckpt experiments/e2e_v7/best --subset first100` — expect RTF ≈ 0.3–0.8 on a modern GPU (Whisper-large-v3 1.55B + LoRA, batch=1, autoregressive).
2. Smoke-run two-stage on GRU 24sess + 5-gram, first100 — expect RTF < 0.5 (Seto reports 78+ WPM for n-gram → well under RTF 1).
3. **Cross-check: WER produced inside the speed run matches EXPERIMENTS.md §3/§2a to ±0.5% absolute** — sanity that we are timing the same pipeline that was evaluated.
4. Confirm `t_neural + t_wfst + t_rescore ≈ t_total` per utterance.
5. Confirm corpus RTF and mean-RTF agree within ~10% (large gap → outlier handling worth reporting).

---

## 8. Out of scope this session (do not get pulled in)

- Probing / cross-attention analysis (EXPERIMENTS.md §5) — separate task.
- Re-running any training or any WER numbers.
- Rebuilding the fine-tuned LLaMA-2 LoRA adapter that was lost (LM7) — different task.
- Anything for the canary eval bug beyond the documented footnote.

---

## 9. Plan file (full)

The full implementation plan from S21 plan mode is at `/root/.claude/plans/take-a-look-at-radiant-charm.md`. It will not survive the machine move — the contents are duplicated above in §3.

---

## 10. After the benchmark runs

Append a **Session 21 entry** to `HANDOFF.md` summarising:
- Hardware used (GPU model, CPU model, free RAM).
- Final §4 numbers from `EXPERIMENTS.md`.
- Any deviations from the protocol above (e.g. canary footnote).
- Whether the WFST `t_wfst` confirms Seto's WPM > 78 claim for n-gram pipelines.

Then delete this file. It's a one-session bridge.
