# Experiments Index

Provenance for every run under `experiments/`. Reconstructed from `wfst_results.json`/`rescore_*.json` configs on 2026-04-14. All full-pipeline runs below use `beam=18, nbest=100, blank_penalty=ln(7)`. "test split of 24-session training data" is the eval split for every row unless noted.

**Convention for new runs:** every new `experiments/<name>/` MUST contain a `README.md` following [_TEMPLATE.md](_TEMPLATE.md) — purpose, exact command, checkpoint + git SHA, hyperparameters, final metrics. `eval_wfst_lm.py` should auto-emit this.

## Phoneme decoder eval only (no LM)

Source: [eval_results.json](eval_results.json).

| Entry | Checkpoint | PER (all sess) |
|---|---|---|
| Conformer Spatial 24sess | `24sess/conformer_spatial_24sess` | 0.1654 |
| Conformer SE 24sess | `24sess/conformer_SE_24sess` (?) | 0.1715 |
| Conformer Vanilla 24sess | `24sess/conformer_vanilla_24sess` | 0.1700 |
| GRU 24sess | (broken — PER=1.0, to rerun) | — |
| Conformer Spatial 19sess | `19sess/conformer/spatial/...` | 0.1839 |
| Conformer SE 19sess | `19sess/conformer/SE/...` | 0.1841 |
| GRU Baseline 19sess | `19sess/gru/baseline/...` | 0.1925 |

## Full-pipeline runs (WFST + optional neural LM rescore)

| Dir | Checkpoint | LM dir | Rescore LM | Grid? | Best asc / α | WER | CER | Oracle WER |
|---|---|---|---|---|---|---|---|---|
| [wfst_lm_5gram](wfst_lm_5gram/) | conformer_spatial_24sess | `speech_5gram/lang_test` | — | no | 0.5 / — | **0.2155** | **0.1466** | 0.1262 |
| [wfst_lm_5gram_asc](wfst_lm_5gram_asc/) | conformer_spatial_24sess | `speech_5gram/lang_test` | — | asc-only | 0.3 / — | **0.2086** | 0.1779 | 0.1262 |
| [wfst_5gram_24sess_conformer_vanilla](wfst_5gram_24sess_conformer_vanilla/) | conformer_vanilla_24sess | `speech_5gram/lang_test` | — | no | 0.5 / — | 0.2170 | 0.1497 | 0.1270 |
| [wfst_5gram_24sess_gru](wfst_5gram_24sess_gru/) | gru_1024u_5L_24sess | `speech_5gram/lang_test` | — | no | 0.5 / — | **0.2141** | 0.1546 | **0.1028** |
| [wfst_lm_gru](wfst_lm_gru/) | gru_1024u_5L_24sess | `languageModel` (3-gram) | — | no | 0.5 / — | 0.2204 | 0.1591 | 0.1033 |
| [wfst_lm](wfst_lm/) | conformer_spatial_24sess | `languageModel` (3-gram) | GPT-2, Gemma-270M | yes | gpt2: 0.3 / 1.2 | 0.2197 (gpt2) / 0.2870 (gemma) | 0.1514 / 0.1765 | 0.1284 |
| [wfst_lm_5gram_asc/rescore_*.json](wfst_lm_5gram_asc/) | conformer_spatial_24sess | `speech_5gram/lang_test` | GPT-2, Gemma-270M, LLaMA-2 7B | yes | gpt2: 0.2/1.2; llama: 0.2/1.2 | 0.2208 / 0.2705 / 0.2405 | 0.1554 / 0.1676 / 0.1570 | 0.1262 |
| [lm_pipeline](lm_pipeline/) | conformer_spatial_24sess | (legacy path, 5-utt smoke test) | — | no | — | 1.53 (smoke) | 0.61 | — |

**Notes on the legacy naming.** `wfst_lm_5gram` and `wfst_lm_5gram_asc` both use the conformer-spatial 24sess checkpoint and the 5-gram LM; the `_asc` variant additionally ran an `asc` grid that found best WER=0.2086 at asc=0.3 (but CER degrades to 0.1779 at that asc). HANDOFF §5 "0.2155" corresponds to the `wfst_lm_5gram` default-asc run and is the honest headline number (asc=0.5, joint WER+CER optimum).

**Rescore caveat.** Every `rescore_*.json` entry above was produced by the current, incomplete fusion `sc = asc*ac + α*lm_neural` in [../AnalysisExamples/rescore_nbest.py](../AnalysisExamples/rescore_nbest.py) — `lm_wfst` is carried in the n-best tuple but **ignored**. So these are not true BSSF results; see plan Step 1.

## N-best files (reusable for rescoring without re-decoding)

- `wfst_lm_5gram_asc/_nbest_tmp.json` — conformer spatial, 5-gram, 787 utts, mean n-best=43
- `wfst_lm/_nbest_tmp.json` — conformer spatial, 3-gram, 779 utts
- `wfst_5gram_24sess_conformer_vanilla/_nbest_tmp.json` — 796 utts, mean n-best=43
- `wfst_5gram_24sess_gru/_nbest_tmp.json` — 846 utts, mean n-best=55

Each file: `{nbest: list[list[[sentence, acoustic_score, wfst_lm_score]]], ground_truth, logit_lengths}`.
