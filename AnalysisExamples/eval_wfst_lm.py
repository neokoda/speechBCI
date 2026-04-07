#!/usr/bin/env python3
"""
WFST Language Model Pipeline Evaluation
========================================
Evaluates our Conformer decoder using Willett et al.'s WFST-based pipeline:
  1. Load Conformer checkpoint → run inference → logits
  2. WFST CTC beam search via lm_decoder (languageModel/TLG.fst, 3-gram)
  3. Optional N-best rescoring with GPT-2 / Gemma 3

This is the Willett/Seto-faithful pipeline. Compare to eval_lm_pipeline.py
which uses our custom Python lexicon-constrained beam search.

Usage:
    python AnalysisExamples/eval_wfst_lm.py [options]
    python AnalysisExamples/eval_wfst_lm.py --lm none            # WFST-only baseline
    python AnalysisExamples/eval_wfst_lm.py --lm gpt2 --alpha 0.5
    python AnalysisExamples/eval_wfst_lm.py --grid-search --lm gpt2
"""

import os
import sys
import json
import time
import argparse
import logging

import numpy as np

# ── GPU / TF setup ──────────────────────────────────────────────────────────
# Re-exec with LD_LIBRARY_PATH pointing at:
#   1. pip-installed NVIDIA libs (TF needs libcudnn.so.8)
#   2. OpenFST build libs (lm_decoder needs libfst.so.8)
# Must be set before the dynamic linker initializes.
import site as _site
_NV_BASE = os.path.join(_site.getsitepackages()[0], 'nvidia')
_FST_LIB = os.path.join(os.path.dirname(__file__), '..',
    'LanguageModelDecoder/runtime/server/x86/fc_base/openfst-build/src/lib/.libs')
_FST_LIB = os.path.normpath(_FST_LIB)
_LM_SO_DIR = os.path.join(os.path.dirname(__file__), '..',
    'LanguageModelDecoder/runtime/server/x86/build/lib.linux-x86_64-cpython-311')
_LM_SO_DIR = os.path.normpath(_LM_SO_DIR)

_extra_libs = [_FST_LIB] if os.path.isdir(_FST_LIB) else []
if os.path.isdir(_NV_BASE):
    _NV_SUBDIRS = ['cudnn', 'cublas', 'cuda_nvrtc', 'cuda_runtime',
                   'cufft', 'cusolver', 'cusparse', 'nvjitlink']
    _extra_libs += [
        os.path.join(_NV_BASE, d, 'lib')
        for d in _NV_SUBDIRS
        if os.path.isdir(os.path.join(_NV_BASE, d, 'lib'))
    ]

_cur_ld = os.environ.get('LD_LIBRARY_PATH', '')
_cur_py = os.environ.get('PYTHONPATH', '')
_need_reexec = (
    any(p not in _cur_ld for p in _extra_libs) or
    (_LM_SO_DIR not in _cur_py and os.path.isdir(_LM_SO_DIR))
)
if _need_reexec:
    _env = os.environ.copy()
    _env['LD_LIBRARY_PATH'] = ':'.join(_extra_libs) + (':' + _cur_ld if _cur_ld else '')
    _env['PYTHONPATH'] = _LM_SO_DIR + (':' + _cur_py if _cur_py else '')
    os.execve(sys.executable, [sys.executable] + sys.argv, _env)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
for _gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu, True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'NeuralDecoder'))
from neuralDecoder.utils.rnnEval import wer as edit_distance
import neuralDecoder.utils.lmDecoderUtils as lmDecoderUtils

# Import inference runner from our existing pipeline script
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
from eval_lm_pipeline import run_inference, FRAME_DURATION_SEC

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
CKPT_DIR    = '/workspace/speechBCI/experiments/24sess/conformer_spatial_24sess'
DATA_DIR    = '/workspace/speechBCI/data/derived/tfRecords'
LM_DIR      = '/workspace/speechBCI/languageModel'
OUTPUT_DIR  = '/workspace/speechBCI/experiments/wfst_lm'

GPT2_MODEL_ID  = 'gpt2'
GEMMA_MODEL_ID = 'google/gemma-3-270m'


# ═══════════════════════════════════════════════════════════════════════════
# Metrics helpers
# ═══════════════════════════════════════════════════════════════════════════

def compute_wer_cer(hypotheses, references, logit_lengths=None):
    """Compute WER, CER, WPM over two lists of strings."""
    total_word_err = total_words = 0
    total_char_err = total_chars = 0
    for hyp, ref in zip(hypotheses, references):
        ref_words = ref.lower().split()
        hyp_words = hyp.lower().split()
        total_word_err += edit_distance(ref_words, hyp_words)
        total_words    += max(len(ref_words), 1)
        total_char_err += edit_distance(list(ref.lower()), list(hyp.lower()))
        total_chars    += max(len(ref), 1)
    wer = total_word_err / total_words if total_words else 0.0
    cer = total_char_err / total_chars if total_chars else 0.0

    wpm = 0.0
    if logit_lengths is not None:
        total_audio_min = float(np.sum(logit_lengths)) * FRAME_DURATION_SEC / 60.0
        gt_words = sum(len(r.split()) for r in references)
        wpm = gt_words / total_audio_min if total_audio_min > 0 else 0.0
    return dict(wer=wer, cer=cer, wpm=wpm)


# ═══════════════════════════════════════════════════════════════════════════
# WFST decoding
# ═══════════════════════════════════════════════════════════════════════════

def build_wfst_decoder(lm_dir, acoustic_scale=0.5, nbest=100, beam=18, load_rescore=False):
    log.info("Loading WFST decoder from %s …", lm_dir)
    if load_rescore:
        log.info("  Lattice rescore FSTs (G.fst + G_no_prune.fst) will be loaded — needs ~80 GB RAM for 5-gram")
    decoder = lmDecoderUtils.build_lm_decoder(
        lm_dir,
        acoustic_scale=acoustic_scale,
        nbest=nbest,
        beam=beam,
        load_rescore=load_rescore,
    )
    log.info("  WFST decoder ready.")
    return decoder


def wfst_decode_all(decoder, logits, logit_lengths, blank_penalty=np.log(7), rescore=False):
    """Run WFST CTC beam search on all utterances.

    Returns list of N-best lists, each entry = (sentence, ac_score, lm_score).
    rescore=True enables WFST lattice rescoring (requires G_no_prune.fst, e.g. 5-gram).
    """
    # Rearrange logits: [blank, SIL, AA..ZH] as expected by tokens.txt
    logits_r = lmDecoderUtils.rearrange_speech_logits(logits, has_sil=True)

    all_nbest = []
    t0 = time.time()
    for i in range(len(logits_r)):
        nbest = lmDecoderUtils.lm_decode(
            decoder,
            logits_r[i, :logit_lengths[i]],
            returnNBest=True,
            blankPenalty=blank_penalty,
            rescore=rescore,
        )
        all_nbest.append(nbest)
        if (i + 1) % 100 == 0:
            log.info("  Decoded %d / %d utts  (%.1f s so far)",
                     i + 1, len(logits_r), time.time() - t0)

    t_total = time.time() - t0
    log.info("  WFST decoding done: %d utts in %.1f s (%.3f s/utt)",
             len(logits_r), t_total, t_total / max(len(logits_r), 1))
    return all_nbest


# (LM rescoring is handled by rescore_nbest.py subprocess — see main())


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def _float_list(s):
    return [float(x) for x in s.split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser(description='WFST LM Pipeline Evaluation')
    parser.add_argument('--lm', choices=['none', 'gpt2', 'gemma', 'both'],
                        default='none',
                        help='Neural LM for N-best rescoring (default: none)')
    parser.add_argument('--beam', type=int, default=18,
                        help='WFST beam width (default: 18, Willett uses 18)')
    parser.add_argument('--nbest', type=int, default=100,
                        help='N-best size from WFST decoder (default: 100)')
    parser.add_argument('--acoustic-scale', type=float, default=0.5,
                        help='Acoustic scale for WFST decoder (default: 0.5)')
    parser.add_argument('--blank-penalty', type=float, default=np.log(7),
                        help='Blank log-penalty for WFST decoder (default: log(7))')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Neural LM weight α for rescoring (default: 0.5)')
    parser.add_argument('--grid-search', action='store_true',
                        help='Grid search over acoustic_scale and alpha')
    parser.add_argument('--alphas', type=_float_list, default=[0.3, 0.5, 0.8, 1.2],
                        help='α values for grid search (comma-separated)')
    parser.add_argument('--acoustic-scales', type=_float_list, default=[0.3, 0.5, 0.8],
                        help='acoustic_scale values for grid search (comma-separated)')
    parser.add_argument('--wfst-rescore', action='store_true',
                        help='Enable WFST lattice rescoring (needs G_no_prune.fst, e.g. 5-gram)')
    parser.add_argument('--max-utts', type=int, default=0,
                        help='Limit to first N utterances (0 = all)')
    parser.add_argument('--ckpt-dir', type=str, default=CKPT_DIR,
                        help='Checkpoint directory (default: conformer_spatial_24sess)')
    parser.add_argument('--lm-dir', type=str, default=LM_DIR)
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--hf-token', type=str, default=None)
    parser.add_argument('--cache-dir', type=str, default='/root/.cache/huggingface')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Conformer inference ──────────────────────────────────────────
    log.info("[1/4] Running Conformer inference …")
    inf = run_inference(ckpt_dir=args.ckpt_dir, data_dir=DATA_DIR)
    logits       = inf['logits']          # (N, T, 41)
    logit_lengths = inf['logitLengths']   # (N,)
    ground_truth  = inf['transcriptions'] # list[str]
    log.info("  Greedy PER = %.4f  |  N=%d utterances", inf['greedy_per'], len(logits))

    if args.max_utts > 0:
        logits        = logits[:args.max_utts]
        logit_lengths = logit_lengths[:args.max_utts]
        ground_truth  = ground_truth[:args.max_utts]
        log.info("  Truncated to %d utterances", args.max_utts)

    # ── Step 2: WFST beam search ─────────────────────────────────────────────
    log.info("[2/4] Loading WFST decoder (TLG.fst ~3-4 min) …")
    decoder = build_wfst_decoder(
        args.lm_dir,
        acoustic_scale=args.acoustic_scale,
        nbest=args.nbest,
        beam=args.beam,
        load_rescore=args.wfst_rescore,
    )

    log.info("[3/4] WFST CTC decoding …")
    if args.wfst_rescore:
        log.info("  WFST lattice rescoring ENABLED (5-gram mode)")
    all_nbest = wfst_decode_all(decoder, logits, logit_lengths,
                                blank_penalty=args.blank_penalty,
                                rescore=args.wfst_rescore)

    # Top-1 from WFST alone
    top1_wfst = [nb[0][0].strip() if nb else '' for nb in all_nbest]
    m_wfst = compute_wer_cer(top1_wfst, ground_truth, logit_lengths)
    log.info("  WFST-only top-1:  WER=%.4f  CER=%.4f  WPM=%.1f",
             m_wfst['wer'], m_wfst['cer'], m_wfst['wpm'])

    # Oracle
    oracle = []
    for i, nb in enumerate(all_nbest):
        ref_words = ground_truth[i].lower().split()
        best_wer, best_s = float('inf'), ''
        for (s, _ac, _lm) in nb:
            w = edit_distance(ref_words, s.lower().split()) / max(len(ref_words), 1)
            if w < best_wer:
                best_wer, best_s = w, s
        oracle.append(best_s)
    m_oracle = compute_wer_cer(oracle, ground_truth, logit_lengths)
    log.info("  ORACLE (best-of-N): WER=%.4f  CER=%.4f", m_oracle['wer'], m_oracle['cer'])

    # ── Step 4: Neural LM rescoring (subprocess to avoid lm_decoder/torch ABI conflict) ──
    # lm_decoder.so links against libtorch 1.13.1; torch 2.5.1 is installed.
    # They cannot coexist in one process. We save N-best to JSON, then call
    # rescore_nbest.py in a fresh subprocess that has no lm_decoder loaded.
    results = {
        'config': vars(args),
        'greedy_per': round(inf['greedy_per'], 6),
        'wfst_only': {k: round(v, 6) for k, v in m_wfst.items()},
        'oracle': {k: round(v, 6) for k, v in m_oracle.items()},
        'lm_results': {},
    }

    lms_to_run = []
    if args.lm in ('gpt2', 'both'):
        lms_to_run.append(('gpt2', GPT2_MODEL_ID))
    if args.lm in ('gemma', 'both'):
        lms_to_run.append(('gemma3_270m', GEMMA_MODEL_ID))

    if lms_to_run:
        import subprocess
        # Serialize N-best and ground-truth for the subprocess
        nbest_path = os.path.join(args.output_dir, '_nbest_tmp.json')
        nbest_serial = [
            [(s, float(ac), float(lm)) for (s, ac, lm) in nb]
            for nb in all_nbest
        ]
        with open(nbest_path, 'w') as f:
            json.dump({'nbest': nbest_serial,
                       'ground_truth': ground_truth,
                       'logit_lengths': logit_lengths.tolist()}, f)
        log.info("[4/4] Neural LM rescoring via subprocess …")

        rescore_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'rescore_nbest.py')
        for lm_tag, lm_id in lms_to_run:
            cmd = [sys.executable, rescore_script,
                   '--nbest-file', nbest_path,
                   '--lm-id', lm_id,
                   '--lm-tag', lm_tag,
                   '--output-dir', args.output_dir,
                   '--cache-dir', args.cache_dir]
            if args.grid_search:
                cmd += ['--grid-search',
                        '--alphas', ','.join(str(a) for a in args.alphas),
                        '--acoustic-scales', ','.join(str(a) for a in args.acoustic_scales)]
            else:
                cmd += ['--alpha', str(args.alpha),
                        '--acoustic-scale', str(args.acoustic_scale)]
            if args.hf_token:
                cmd += ['--hf-token', args.hf_token]

            log.info("  Launching rescorer for %s …", lm_tag)
            proc = subprocess.run(cmd, capture_output=False)
            if proc.returncode != 0:
                log.error("  Rescorer failed for %s (exit %d)", lm_tag, proc.returncode)
                continue

            rescore_out = os.path.join(args.output_dir, f'rescore_{lm_tag}.json')
            if os.path.exists(rescore_out):
                with open(rescore_out) as f:
                    results['lm_results'][lm_tag] = json.load(f)
            else:
                log.warning("  No output file found: %s", rescore_out)

    # ── Save results ─────────────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir, 'wfst_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", out_path)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  WFST LM Pipeline Results")
    print("="*60)
    print(f"  Greedy PER          : {inf['greedy_per']:.4f}")
    print(f"  WFST-only top-1 WER : {m_wfst['wer']:.4f}  CER: {m_wfst['cer']:.4f}")
    print(f"  Oracle WER          : {m_oracle['wer']:.4f}  CER: {m_oracle['cer']:.4f}")
    for lm_tag, lm_res in results['lm_results'].items():
        if 'best' in lm_res:
            b = lm_res['best']
            print(f"  {lm_tag} (best)    : WER={b['wer']:.4f}  CER={b['cer']:.4f}")
        elif 'wer' in lm_res:
            print(f"  {lm_tag}            : WER={lm_res['wer']:.4f}  CER={lm_res['cer']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
