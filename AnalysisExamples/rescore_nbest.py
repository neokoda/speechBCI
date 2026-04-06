#!/usr/bin/env python3
"""
N-best Rescoring with Neural LM
================================
Standalone script that runs GPT-2 or Gemma rescoring on WFST N-best hypotheses.
Called as a subprocess from eval_wfst_lm.py to avoid lm_decoder/torch ABI conflict
(lm_decoder links against libtorch 1.13.1; torch 2.5.1 cannot coexist in same process).

Input:  --nbest-file  JSON with {nbest, ground_truth, logit_lengths}
Output: rescore_<lm_tag>.json in --output-dir
"""

import os
import sys
import json
import argparse
import logging
import time

import numpy as np

# ── GPU setup (torch only, no TF) ──────────────────────────────────────────
# Re-exec with correct LD_LIBRARY_PATH so CUDA libs are found.
import site as _site
_NV_BASE = os.path.join(_site.getsitepackages()[0], 'nvidia')
if os.path.isdir(_NV_BASE):
    _NV_SUBDIRS = ['cudnn', 'cublas', 'cuda_nvrtc', 'cuda_runtime',
                   'cufft', 'cusolver', 'cusparse', 'nvjitlink']
    _NV_LIBS = ':'.join(
        os.path.join(_NV_BASE, d, 'lib')
        for d in _NV_SUBDIRS
        if os.path.isdir(os.path.join(_NV_BASE, d, 'lib'))
    )
    _cur_ld = os.environ.get('LD_LIBRARY_PATH', '')
    if _NV_LIBS and _NV_LIBS.split(':')[0] not in _cur_ld:
        _env = os.environ.copy()
        _env['LD_LIBRARY_PATH'] = _NV_LIBS + ':' + _cur_ld
        os.execve(sys.executable, [sys.executable] + sys.argv, _env)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'NeuralDecoder'))
from neuralDecoder.utils.rnnEval import wer as edit_distance

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

FRAME_DURATION_SEC = 0.080


def load_lm(model_id, cache_dir=None, hf_token=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info("  Loading %s on %s …", model_id, device)
    kwargs = {'cache_dir': cache_dir, 'low_cpu_mem_usage': True}
    if hf_token:
        kwargs['token'] = hf_token
    if 'gemma' in model_id.lower():
        kwargs['torch_dtype'] = torch.bfloat16
        kwargs['trust_remote_code'] = True
    else:
        kwargs['torch_dtype'] = torch.float16
    tok_kwargs = {'cache_dir': cache_dir}
    if hf_token:
        tok_kwargs['token'] = hf_token
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(device).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def rescore_with_lm(model, tokenizer, hypotheses, length_penalty=0.0):
    """Return log-prob scores for each hypothesis string."""
    model_is_tf = type(model).__name__.startswith('TF')
    if model_is_tf:
        raise RuntimeError("rescore_nbest.py expects a PyTorch model")

    device = next(model.parameters()).device
    inputs = tokenizer(hypotheses, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        log_probs = torch.nn.functional.log_softmax(outputs.logits.float(), -1).cpu().numpy()

    scores = []
    attn = inputs['attention_mask'].cpu().numpy()
    ids  = inputs['input_ids'].cpu().numpy()
    for i in range(len(hypotheses)):
        n_tok = int(attn[i].sum())
        score = sum(log_probs[i, j-1, ids[i, j]] for j in range(1, n_tok))
        scores.append(float(score - n_tok * length_penalty))
    return scores


def compute_wer_cer(hypotheses, references, logit_lengths=None):
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


def _float_list(s):
    return [float(x) for x in s.split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nbest-file', required=True)
    parser.add_argument('--lm-id', required=True)
    parser.add_argument('--lm-tag', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--acoustic-scale', type=float, default=0.5)
    parser.add_argument('--grid-search', action='store_true')
    parser.add_argument('--alphas', type=_float_list, default=[0.3, 0.5, 0.8, 1.2])
    parser.add_argument('--acoustic-scales', type=_float_list, default=[0.3, 0.5, 0.8])
    parser.add_argument('--hf-token', type=str, default=None)
    parser.add_argument('--cache-dir', type=str, default='/root/.cache/huggingface')
    args = parser.parse_args()

    with open(args.nbest_file) as f:
        data = json.load(f)
    all_nbest     = data['nbest']        # list of list of [s, ac, lm]
    ground_truth  = data['ground_truth']
    logit_lengths = np.array(data['logit_lengths'])

    model, tokenizer = load_lm(args.lm_id, cache_dir=args.cache_dir,
                                hf_token=args.hf_token or None)

    # Score all hypotheses
    log.info("Rescoring %d utterances …", len(all_nbest))
    t0 = time.time()
    enriched = []
    for i, nbest in enumerate(all_nbest):
        hyps = [h[0].strip() for h in nbest if h[0].strip()]
        if not hyps:
            enriched.append([])
            continue
        lm_scores = rescore_with_lm(model, tokenizer, hyps)
        entry = []
        j = 0
        for h in nbest:
            s = h[0].strip()
            if not s:
                continue
            entry.append((s, float(h[1]), float(h[2]), lm_scores[j]))
            j += 1
        enriched.append(entry)
        if (i + 1) % 100 == 0:
            log.info("  %d / %d  (%.1f s)", i + 1, len(all_nbest), time.time() - t0)
    log.info("Rescoring done in %.1f s", time.time() - t0)

    def pick_best(asc, alpha):
        decoded = []
        for nb in enriched:
            if not nb:
                decoded.append('')
                continue
            best_s, best_sc = '', float('-inf')
            for (s, ac, lm_wfst, lm_neural) in nb:
                sc = asc * ac + alpha * lm_neural
                if sc > best_sc:
                    best_sc, best_s = sc, s
            decoded.append(best_s)
        return decoded

    out = {}
    if args.grid_search:
        best = None
        grid = []
        for asc in args.acoustic_scales:
            for a in args.alphas:
                decoded = pick_best(asc, a)
                m = compute_wer_cer(decoded, ground_truth, logit_lengths)
                grid.append({'acoustic_scale': asc, 'alpha': a,
                             'wer': round(m['wer'], 6), 'cer': round(m['cer'], 6)})
                log.info("  asc=%.2f  α=%.2f → WER=%.4f  CER=%.4f",
                         asc, a, m['wer'], m['cer'])
                if best is None or m['wer'] < best['wer']:
                    best = grid[-1]
        log.info("Best: asc=%.2f  α=%.2f → WER=%.4f  CER=%.4f",
                 best['acoustic_scale'], best['alpha'], best['wer'], best['cer'])
        out = {'grid': grid, 'best': best}
    else:
        decoded = pick_best(args.acoustic_scale, args.alpha)
        m = compute_wer_cer(decoded, ground_truth, logit_lengths)
        log.info("%s: WER=%.4f  CER=%.4f  WPM=%.1f",
                 args.lm_tag, m['wer'], m['cer'], m['wpm'])
        out = {k: round(v, 6) for k, v in m.items()}

    out_path = os.path.join(args.output_dir, f'rescore_{args.lm_tag}.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    log.info("Saved to %s", out_path)


if __name__ == '__main__':
    main()
