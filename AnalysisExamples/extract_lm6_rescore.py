#!/usr/bin/env python3
"""Re-run base LLaMA-2-7B n-best rescoring (LM6) on the Conformer-spatial 5-gram
n-best cache and save PER-UTTERANCE output + fused-score confidence features for
the ensembling experiments.

The existing AnalysisExamples/rescore_nbest.py only writes corpus WER/CER, so the
per-utterance LM6 hypotheses and their fusion scores were never cached. This
script reuses the same scoring (rescore_with_lm) and the same best fusion config
(asc=0.5, alpha=1.2, beta=1.0, gamma=0.0 — from experiments/bssf_5gram_llama2_
conformer_spatial/bssf_llama2_7b.json) and dumps everything needed by the router.

Fusion (BSSF log-linear): sc = asc*ac + beta*lm_wfst + alpha*lm_neural + gamma*len.

Weights: NousResearch/Llama-2-7b-hf — an ungated, bit-identical re-upload of
meta-llama/Llama-2-7b-hf (gated, no HF token here), so scores match the thesis.

Output (aligned 1:1, 880 utts): experiments/analysis/lm6_llama2_rescore.json
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_lm(model_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_id} on {device} …")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # The scoring loop sums log P over indices [1, n_tok), which assumes real
    # tokens are LEFT-aligned (right padding). This tokenizer defaults to
    # left-padding under transformers 5.x, which would score pad tokens and
    # yield garbage LM scores — force right padding.
    tok.padding_side = "right"
    return model, tok


@torch.no_grad()
def rescore_with_lm(model, tokenizer, hypotheses, chunk_size=32):
    """Sum of token log-probs (LM log-likelihood) for each hypothesis string."""
    device = next(model.parameters()).device
    scores = []
    for start in range(0, len(hypotheses), chunk_size):
        chunk = hypotheses[start:start + chunk_size]
        inputs = tokenizer(chunk, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits
        log_probs = torch.nn.functional.log_softmax(logits.float(), -1).cpu().numpy()
        attn = inputs["attention_mask"].cpu().numpy()
        ids = inputs["input_ids"].cpu().numpy()
        for i in range(len(chunk)):
            n_tok = int(attn[i].sum())
            score = sum(log_probs[i, j - 1, ids[i, j]] for j in range(1, n_tok))
            scores.append(float(score))
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nbest-file",
                   default="experiments/wfst_lm_5gram_asc/_nbest_tmp.json")
    p.add_argument("--model-id", default="NousResearch/Llama-2-7b-hf")
    p.add_argument("--output", default="experiments/analysis/lm6_llama2_rescore.json")
    p.add_argument("--asc", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=1.2)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--chunk-size", type=int, default=32)
    args = p.parse_args()

    with open(args.nbest_file) as f:
        data = json.load(f)
    all_nbest = data["nbest"]
    gt = data["ground_truth"]
    logit_lengths = data["logit_lengths"]
    print(f"Loaded {len(all_nbest)} utterances")

    model, tok = load_lm(args.model_id)

    rows = []
    t0 = time.time()
    tot_e = tot_w = 0
    for i, nbest in enumerate(all_nbest):
        hyps = [h[0].strip() for h in nbest if h[0].strip()]
        if not hyps:
            rows.append({"ref": gt[i], "hyp_lm6": "", "logit_len": logit_lengths[i],
                         "fused": [], "hyps": []})
            continue
        lm_neural = rescore_with_lm(model, tok, hyps, chunk_size=args.chunk_size)
        fused, kept = [], []
        j = 0
        for h in nbest:
            s = h[0].strip()
            if not s:
                continue
            ac, lm_wfst, neu = float(h[1]), float(h[2]), lm_neural[j]
            n = len(s.split())
            sc = args.asc * ac + args.beta * lm_wfst + args.alpha * neu + args.gamma * n
            fused.append(sc)
            kept.append({"text": s, "ac": ac, "lm_wfst": lm_wfst,
                         "lm_neural": neu, "fused": sc})
            j += 1
        top = int(np.argmax(fused))
        hyp_lm6 = kept[top]["text"]
        rows.append({
            "ref": gt[i], "hyp_lm6": hyp_lm6, "logit_len": logit_lengths[i],
            "hyps": kept,
        })
        # running WER (word-level, lowercase) as a sanity check
        rw, hw = gt[i].lower().split(), hyp_lm6.lower().split()
        d = list(range(len(hw) + 1))
        for a_ in rw:
            prev, d[0] = d[0], d[0] + 1
            for k, b_ in enumerate(hw, 1):
                cur = d[k]; d[k] = min(d[k] + 1, d[k - 1] + 1, prev + (a_ != b_)); prev = cur
        tot_e += d[-1]; tot_w += max(1, len(rw))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(all_nbest)}  ({time.time()-t0:.0f}s)  "
                  f"running WER={tot_e/max(1,tot_w):.4f}")

    print(f"all_24 WER (sanity) = {tot_e/max(1,tot_w):.4f}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"model_id": args.model_id,
                   "config": {"asc": args.asc, "alpha": args.alpha,
                              "beta": args.beta, "gamma": args.gamma},
                   "n": len(rows), "rows": rows}, f)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
