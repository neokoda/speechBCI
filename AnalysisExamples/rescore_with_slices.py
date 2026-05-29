#!/usr/bin/env python3
"""Rescore N-best with LLaMA-2 7B (base), grid-search fusion, save WER/CER per slice."""
import argparse, json, os, sys, time

import torch
import numpy as np

# Re-exec with LD_LIBRARY_PATH if needed (matches rescore_nbest.py header)
import site as _site
_NV_BASE = os.path.join(_site.getsitepackages()[0], 'nvidia')
if os.path.isdir(_NV_BASE):
    _NV_SUBDIRS = ['cudnn','cublas','cuda_nvrtc','cuda_runtime','cufft','cusolver','cusparse','nvjitlink']
    _NV_LIBS = ':'.join(os.path.join(_NV_BASE, d, 'lib') for d in _NV_SUBDIRS if os.path.isdir(os.path.join(_NV_BASE, d, 'lib')))
    _cur = os.environ.get('LD_LIBRARY_PATH','')
    if _NV_LIBS and _NV_LIBS.split(':')[0] not in _cur:
        e=os.environ.copy(); e['LD_LIBRARY_PATH']=_NV_LIBS+':'+_cur
        os.execve(sys.executable, [sys.executable]+sys.argv, e)

from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from e2e.eval import ALL_SESSIONS, WILLETT_19, WILLETT_4_18

SLICE_SETS = {
    'all_24':       set(range(len(ALL_SESSIONS))),
    'willett_19':   {ALL_SESSIONS.index(s) for s in WILLETT_19},
    'willett_4_18': {ALL_SESSIONS.index(s) for s in WILLETT_4_18},
}


def load_session_idx_for_utts(data_dir, sessions):
    """Run BCIDataset once to get session_idx per test utterance, in BCIDataset order
    (which we verified matches the WFST nbest order 880/880)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from e2e.dataset import BCIDataset
    ds = BCIDataset(data_dir, sessions, split='test', tokenizer=None, max_text_len=64, augment=False)
    return [ex['session_idx'] for ex in ds.examples], [ex['text'].strip() for ex in ds.examples]


def edit(a, b):
    d = list(range(len(b)+1))
    for ai in a:
        nd = [d[0]+1]
        for j, bi in enumerate(b):
            nd.append(min(d[j+1]+1, nd[j]+1, d[j]+(0 if ai==bi else 1)))
        d = nd
    return d[len(b)]


def slice_wcer(hyps, refs, sess_idx, slice_ids):
    tot_w = err_w = tot_c = err_c = n = 0
    for i, (h, r) in enumerate(zip(hyps, refs)):
        if sess_idx[i] not in slice_ids:
            continue
        rl, hl = r.lower(), h.lower()
        rw, hw = rl.split(), hl.split()
        tot_w += len(rw); err_w += edit(rw, hw)
        tot_c += len(rl); err_c += edit(rl, hl)
        n += 1
    return err_w/max(1,tot_w), err_c/max(1,tot_c), n


def rescore_hyps(model, tokenizer, hyps, length_penalty=0.0, chunk_size=32):
    scores = []
    n = len(hyps)
    for i in range(0, n, chunk_size):
        batch = hyps[i:i+chunk_size]
        enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            ids = enc['input_ids']
            attn = enc['attention_mask']
            for j in range(len(batch)):
                n_tok = int(attn[j].sum().item())
                # P(t_k | t_<k); skip BOS at j=0
                s = 0.0
                for k in range(1, n_tok):
                    s += log_probs[j, k-1, ids[j, k]].item()
                scores.append(s - n_tok * length_penalty)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nbest-file', required=True)
    ap.add_argument('--output', required=True, help='output JSON path')
    ap.add_argument('--data-dir', default='data/derived/tfRecords')
    ap.add_argument('--lm-id', default='meta-llama/Llama-2-7b-hf')
    ap.add_argument('--cache-dir', default='/workspace/.hf_home')
    ap.add_argument('--hf-token', default=None)
    ap.add_argument('--alphas', type=lambda s: [float(x) for x in s.split(',')], default=[0.0, 0.3, 0.5, 0.8, 1.2])
    ap.add_argument('--betas',  type=lambda s: [float(x) for x in s.split(',')], default=[0.5, 1.0, 1.5])
    ap.add_argument('--gammas', type=lambda s: [float(x) for x in s.split(',')], default=[0.0])
    ap.add_argument('--ascs',   type=lambda s: [float(x) for x in s.split(',')], default=[0.3, 0.5, 0.7])
    args = ap.parse_args()

    print(f'Loading nbest from {args.nbest_file}')
    with open(args.nbest_file) as f: data = json.load(f)
    all_nbest = data['nbest']
    gt = [s.strip() for s in data['ground_truth']]
    print(f'  n utts: {len(all_nbest)}')

    print(f'Loading session_idx ordering via BCIDataset')
    sess_idx, ds_gt = load_session_idx_for_utts(args.data_dir, ALL_SESSIONS)
    # When the caller passes the FULL dataset n-best, validate alignment against
    # BCIDataset order. For a SUBSET (speed-benchmark mode, e.g. willett_4_18 which
    # is non-contiguous), the subset's own ground_truth is already aligned to its
    # n-best by the caller, so skip the full-dataset alignment + slice-WER grid.
    is_subset = len(all_nbest) != len(ds_gt)
    if not is_subset:
        assert all(a.lower() == b.lower() for a,b in zip(gt, ds_gt)), 'order mismatch!'

    print(f'Loading LLaMA-2 7B from {args.lm_id}')
    tok = AutoTokenizer.from_pretrained(args.lm_id, cache_dir=args.cache_dir, token=args.hf_token)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.lm_id, cache_dir=args.cache_dir, token=args.hf_token,
        torch_dtype=torch.float16, device_map='cuda',
    )
    model.eval()

    print('Rescoring all hypotheses')
    t0 = time.time()
    enriched = []
    for i, nb in enumerate(all_nbest):
        hyps = [h[0].strip() for h in nb if h[0].strip()]
        if not hyps:
            enriched.append([]); continue
        lm = rescore_hyps(model, tok, hyps)
        entry = []
        j = 0
        for h in nb:
            s = h[0].strip()
            if not s: continue
            entry.append((s, float(h[1]), float(h[2]), float(lm[j])))
            j += 1
        enriched.append(entry)
        if (i+1) % 100 == 0:
            print(f'  {i+1}/{len(all_nbest)}  ({time.time()-t0:.1f}s)')
    t_rescore_pure = time.time() - t0   # pure LLaMA forward time, EXCLUDES model load
    print(f'Rescoring done in {t_rescore_pure:.1f}s')

    # Speed-benchmark (subset) mode: only the per-utterance rescore time matters.
    # Skip the slice-WER grid search (assumes full dataset ordering) and write timing.
    if is_subset:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({'t_rescore_pure_s': t_rescore_pure,
                       'n_utts': len(all_nbest)}, f, indent=2)
        print(f'[subset mode] Saved rescore timing → {args.output}')
        return

    def pick_best(asc, alpha, beta, gamma):
        out = []
        for nb in enriched:
            if not nb: out.append(''); continue
            best_s, best_sc = '', float('-inf')
            for s, ac, lm_w, lm_n in nb:
                n = len(s.split())
                sc = asc*ac + beta*lm_w + alpha*lm_n + gamma*n
                if sc > best_sc: best_sc, best_s = sc, s
            out.append(best_s)
        return out

    grid = []
    best_per_slice = {s: None for s in SLICE_SETS}
    print('\nGrid search × slice')
    for asc in args.ascs:
        for b in args.betas:
            for a in args.alphas:
                for g in args.gammas:
                    hyps = pick_best(asc, a, b, g)
                    slices = {}
                    for sname, sids in SLICE_SETS.items():
                        w, c, n = slice_wcer(hyps, gt, sess_idx, sids)
                        slices[sname] = {'wer': round(w,6), 'cer': round(c,6), 'n': n}
                    row = {'asc': asc, 'alpha': a, 'beta': b, 'gamma': g, 'slices': slices}
                    grid.append(row)
                    print(f"  asc={asc:.1f} β={b:.1f} α={a:.1f} γ={g:.1f}  "
                          f"WER all_24={slices['all_24']['wer']:.4f}  "
                          f"willett_19={slices['willett_19']['wer']:.4f}  "
                          f"willett_4_18={slices['willett_4_18']['wer']:.4f}")
                    for sname in SLICE_SETS:
                        if best_per_slice[sname] is None or slices[sname]['wer'] < best_per_slice[sname]['slices'][sname]['wer']:
                            best_per_slice[sname] = row

    print('\nBest per slice:')
    for sname, r in best_per_slice.items():
        s = r['slices'][sname]
        print(f"  {sname}: WER={s['wer']:.4f} CER={s['cer']:.4f} n={s['n']}  "
              f"(asc={r['asc']:.1f} β={r['beta']:.1f} α={r['alpha']:.1f} γ={r['gamma']:.1f})")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({'grid': grid, 'best_per_slice': best_per_slice,
                   't_rescore_pure_s': t_rescore_pure}, f, indent=2)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
