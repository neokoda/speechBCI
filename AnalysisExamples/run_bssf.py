#!/usr/bin/env python3
"""
Standalone BSSF rescoring driver for an already-saved N-best file.

Skips Conformer inference + WFST decoding entirely — works directly on
`_nbest_tmp.json` produced by a prior eval_wfst_lm.py run. This makes neural-LM
rescoring experiments cheap (minutes, not hours) and avoids the lm_decoder/torch
ABI conflict (no lm_decoder import needed).

Invokes `rescore_nbest.py` as a subprocess for each LM with the requested BSSF
grid. Writes `bssf_<lm_tag>.json` and a per-experiment README.md with the exact
command, git SHA, and best config.

Usage:
    python AnalysisExamples/run_bssf.py \\
        --nbest-file experiments/wfst_lm_5gram_asc/_nbest_tmp.json \\
        --lm gpt2 \\
        --output-dir experiments/bssf_5gram_gpt2 \\
        --grid-search
"""
import argparse
import json
import os
import subprocess
import sys
import time
import datetime as _dt


LM_IDS = {
    'gpt2':        'gpt2',
    'gemma3_270m': 'google/gemma-3-270m',
    'llama2_7b':   'meta-llama/Llama-2-7b-hf',
}


def _float_list(s):
    return [float(x) for x in s.split(',') if x.strip()]


def git_sha():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def write_readme(out_dir, purpose, cmd, cfg, results):
    lines = [
        f"# BSSF Rescore: {cfg['lm_tag']} on {os.path.basename(cfg['nbest_file'])}",
        '',
        f"**Purpose:** {purpose}",
        '',
        f"**Date:** {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Git SHA:** `{cfg['git_sha']}`",
        '',
        '## Command',
        '',
        '```bash',
        cmd,
        '```',
        '',
        '## Configuration',
        '',
        f"- **N-best source:** `{cfg['nbest_file']}`",
        f"- **LM:** `{cfg['lm_id']}` (tag `{cfg['lm_tag']}`)",
        f"- **Fusion:** `sc = asc*ac + beta*lm_wfst + alpha*lm_neural + gamma*len(s.split())`",
        f"- **Grid:** asc={cfg['acoustic_scales']}  beta={cfg['betas']}  alpha={cfg['alphas']}  gamma={cfg['gammas']}",
        '',
        '## Results',
        '',
    ]
    if 'best' in results:
        b = results['best']
        lines += [
            f"**Best:** asc={b['acoustic_scale']}  β={b['beta']}  α={b['alpha']}  γ={b['gamma']}  →  **WER={b['wer']:.4f}  CER={b['cer']:.4f}**",
            '',
            '### Full grid',
            '',
            '| asc | β | α | γ | WER | CER |',
            '|---|---|---|---|---|---|',
        ]
        for g in results['grid']:
            lines.append(f"| {g['acoustic_scale']} | {g['beta']} | {g['alpha']} | {g['gamma']} | {g['wer']:.4f} | {g['cer']:.4f} |")
    else:
        lines.append(f"WER={results.get('wer','?')}  CER={results.get('cer','?')}")
    lines += ['', '## Notes', '', '(add follow-ups or caveats)']
    with open(os.path.join(out_dir, 'README.md'), 'w') as f:
        f.write('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nbest-file', required=True)
    ap.add_argument('--lm', required=True, choices=list(LM_IDS.keys()),
                    help='which neural LM to rescore with')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--grid-search', action='store_true')
    ap.add_argument('--alphas', type=_float_list, default=[0.0, 0.3, 0.5, 0.8, 1.2])
    ap.add_argument('--betas',  type=_float_list, default=[0.5, 1.0, 1.5])
    ap.add_argument('--gammas', type=_float_list, default=[0.0])
    ap.add_argument('--acoustic-scales', type=_float_list, default=[0.3, 0.5, 0.7])
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--beta',  type=float, default=1.0)
    ap.add_argument('--gamma', type=float, default=0.0)
    ap.add_argument('--acoustic-scale', type=float, default=0.5)
    ap.add_argument('--hf-token', type=str, default=None)
    ap.add_argument('--cache-dir', type=str, default='/root/.cache/huggingface')
    ap.add_argument('--lora-dir', type=str, default=None,
                    help='optional PEFT LoRA adapter dir to attach to the base LM')
    ap.add_argument('--purpose', type=str, default='BSSF rescoring experiment')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    lm_id = LM_IDS[args.lm]

    here = os.path.dirname(os.path.abspath(__file__))
    rescore_py = os.path.join(here, 'rescore_nbest.py')

    cmd = [sys.executable, rescore_py,
           '--nbest-file', args.nbest_file,
           '--lm-id',  lm_id,
           '--lm-tag', args.lm,
           '--output-dir', args.output_dir,
           '--cache-dir', args.cache_dir]
    if args.grid_search:
        cmd += ['--grid-search',
                '--alphas', ','.join(str(x) for x in args.alphas),
                '--betas',  ','.join(str(x) for x in args.betas),
                '--gammas', ','.join(str(x) for x in args.gammas),
                '--acoustic-scales', ','.join(str(x) for x in args.acoustic_scales)]
    else:
        cmd += ['--alpha', str(args.alpha), '--beta', str(args.beta),
                '--gamma', str(args.gamma), '--acoustic-scale', str(args.acoustic_scale)]
    if args.hf_token:
        cmd += ['--hf-token', args.hf_token]
    if args.lora_dir:
        cmd += ['--lora-dir', args.lora_dir]

    # Redact the HF token from anything we print or save to README (prevents leaks into git).
    safe_cmd = [('$HF_TOKEN' if (i > 0 and cmd[i-1] == '--hf-token') else v)
                for i, v in enumerate(cmd)]
    print('Running:', ' '.join(safe_cmd), flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    dt = time.time() - t0
    print(f'Rescorer exited {rc} after {dt:.1f}s', flush=True)
    if rc != 0:
        sys.exit(rc)

    # rescore_nbest.py writes rescore_<tag>.json — we adopt that as bssf_<tag>.json for clarity
    src = os.path.join(args.output_dir, f'rescore_{args.lm}.json')
    dst = os.path.join(args.output_dir, f'bssf_{args.lm}.json')
    if os.path.exists(src):
        os.replace(src, dst)
        with open(dst) as f:
            results = json.load(f)
    else:
        results = {}

    cfg = {
        'lm_tag': args.lm, 'lm_id': lm_id,
        'nbest_file': args.nbest_file,
        'acoustic_scales': args.acoustic_scales if args.grid_search else [args.acoustic_scale],
        'betas':  args.betas  if args.grid_search else [args.beta],
        'alphas': args.alphas if args.grid_search else [args.alpha],
        'gammas': args.gammas if args.grid_search else [args.gamma],
        'git_sha': git_sha(),
    }
    write_readme(args.output_dir, args.purpose, ' '.join(safe_cmd), cfg, results)
    print(f'README written to {args.output_dir}/README.md')


if __name__ == '__main__':
    main()
