#!/usr/bin/env python3
"""Extract per-utterance confidence (sequence log-prob) from the E2E
Whisper-large-v3 checkpoint (e2e_v7) for the ensembling experiments.

Mirrors the whisper branch of e2e/eval.py exactly (same loader, shuffle=False,
same session order) so the output rows align 1:1 by index with
experiments/e2e_v7/eval_full.json["details"] and with the two-stage n-best
cache (experiments/wfst_lm_5gram_asc/_nbest_tmp.json).

For each test utterance it records:
  - ref, hyp, session_idx
  - sum_logprob   : sum of per-token log P (greedy, length-unnormalised)
  - n_tokens      : number of generated (non-pad) tokens
  - mean_logprob  : sum_logprob / n_tokens (length-normalised confidence)

Run from the AnalysisExamples directory:
  cd AnalysisExamples
  python extract_e2e_confidence.py \
      --data-dir ../data/derived/tfRecords \
      --ckpt ../experiments/e2e_v7/best \
      --output ../experiments/analysis/e2e_v7_confidence.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from e2e.dataset import BCIDataset, bci_collate_fn
from e2e.whisper_model import WhisperBCIModel

ALL_SESSIONS = [
    "t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17", "t12.2022.05.19",
    "t12.2022.05.24", "t12.2022.05.26", "t12.2022.06.02", "t12.2022.06.07",
    "t12.2022.06.14", "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.23",
    "t12.2022.06.28", "t12.2022.07.05", "t12.2022.07.14", "t12.2022.07.21",
    "t12.2022.07.27", "t12.2022.07.29", "t12.2022.08.02", "t12.2022.08.11",
    "t12.2022.08.13", "t12.2022.08.18", "t12.2022.08.23", "t12.2022.08.25",
]


@torch.no_grad()
def generate_with_scores(model, ecog, ecog_len, session_ids, tokenizer,
                         max_new_tokens=64):
    """Greedy generate (num_beams=1) returning (texts, sum_lp, n_tok)."""
    ecog_memory, enc_len = model._encode_ecog(ecog, ecog_len, session_ids)
    B, T_prime, _ = ecog_memory.shape
    ecog_mask = model._ecog_mask(B, T_prime, enc_len, ecog.device)

    lang_kwargs = {"language": "en"} if model._is_multilingual else {}
    out = model.whisper.generate(
        encoder_outputs=BaseModelOutput(last_hidden_state=ecog_memory),
        attention_mask=ecog_mask,
        max_new_tokens=max_new_tokens,
        max_length=None,
        num_beams=1,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        **lang_kwargs,
    )
    seqs = out.sequences                                   # (B, prompt+gen)
    # Per-token log-probs for the generated portion (normalize_logits=True
    # turns raw logits into log-softmax probabilities).
    hf = model.whisper.get_base_model() if hasattr(model.whisper, "get_base_model") else model.whisper
    trans = hf.compute_transition_scores(
        seqs, out.scores, normalize_logits=True
    )                                                      # (B, gen_len)
    gen_len = trans.shape[1]
    gen_tokens = seqs[:, -gen_len:]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    mask = (gen_tokens != pad_id)
    # compute_transition_scores yields 0.0 (not -inf) for padded steps of
    # finished sequences, but mask defensively so they never contribute.
    trans = torch.where(mask, trans, torch.zeros_like(trans))
    sum_lp = trans.sum(dim=1)                              # (B,)
    n_tok = mask.sum(dim=1).clamp(min=1)                   # (B,)

    texts = tokenizer.batch_decode(seqs, skip_special_tokens=True)
    texts = [t.strip() for t in texts]
    return texts, sum_lp.float().cpu().tolist(), n_tok.cpu().tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--whisper-model", default="openai/whisper-large-v3")
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-text-len", type=int, default=64)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.whisper_model, use_fast=True)
    sessions = ALL_SESSIONS

    train_ds = BCIDataset(args.data_dir, sessions, split="train",
                          tokenizer=tokenizer, max_text_len=args.max_text_len,
                          augment=False)
    session_stats = train_ds.get_session_stats_for_model()

    ds = BCIDataset(args.data_dir, sessions, split="test",
                    tokenizer=tokenizer, max_text_len=args.max_text_len,
                    augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=bci_collate_fn, num_workers=0)
    print(f"Test examples: {len(ds)}")

    model = WhisperBCIModel(
        whisper_name=args.whisper_model,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        cross_attn_only=False, n_sessions=len(sessions),
    )
    model.build_per_session_norm(session_stats)
    ckpt = torch.load(os.path.join(args.ckpt, "checkpoint.pt"),
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint at step {ckpt.get('step', '?')}")

    rows = []
    for i, batch in enumerate(loader):
        ecog = batch["ecog"].to(device)
        ecog_len = batch["ecog_lengths"].to(device)
        session_ids = batch["session_idx"].to(device)
        texts, sum_lp, n_tok = generate_with_scores(
            model, ecog, ecog_len, session_ids, tokenizer,
            max_new_tokens=args.max_new_tokens,
        )
        for h, ref, sidx, slp, nt in zip(
            texts, batch["texts"], session_ids.cpu().tolist(), sum_lp, n_tok
        ):
            rows.append({
                "ref": ref,
                "hyp": h,
                "session_idx": int(sidx),
                "sum_logprob": float(slp),
                "n_tokens": int(nt),
                "mean_logprob": float(slp) / max(1, int(nt)),
            })
        if (i + 1) % 20 == 0:
            print(f"  batch {i+1}/{len(loader)}  ({len(rows)} utts)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"checkpoint": args.ckpt, "n": len(rows), "rows": rows}, f)
    print(f"Wrote {args.output}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
