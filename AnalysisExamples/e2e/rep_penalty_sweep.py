#!/usr/bin/env python3
"""Repetition penalty sweep — find the optimal value for greedy decoding.

Loads a checkpoint and runs validation with each repetition_penalty value
in --penalties. Reports WER for each. The optimal value minimizes WER.

Usage:
    python AnalysisExamples/e2e/rep_penalty_sweep.py \\
        --data-dir data/derived/tfRecords \\
        --ckpt experiments/e2e_v4/best \\
        --lm Qwen/Qwen3.5-0.8B-Base \\
        --penalties 1.0 1.05 1.1 1.15 1.2 1.3 \\
        --batch-size 8 \\
        --output rep_penalty_sweep.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from e2e.model import E2EBCIModel
from e2e.dataset import BCIDataset, bci_collate_fn

ALL_SESSIONS = [
    "t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17", "t12.2022.05.19",
    "t12.2022.05.24", "t12.2022.05.26", "t12.2022.06.02", "t12.2022.06.07",
    "t12.2022.06.14", "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.23",
    "t12.2022.06.28", "t12.2022.07.05", "t12.2022.07.14", "t12.2022.07.21",
    "t12.2022.07.27", "t12.2022.07.29", "t12.2022.08.02", "t12.2022.08.11",
    "t12.2022.08.13", "t12.2022.08.18", "t12.2022.08.23", "t12.2022.08.25",
]


def _edit_distance(a, b):
    d = list(range(len(b) + 1))
    for ai in a:
        nd = [d[0] + 1]
        for j, bi in enumerate(b):
            nd.append(min(d[j+1]+1, nd[j]+1, d[j] + (0 if ai == bi else 1)))
        d = nd
    return d[len(b)]


def compute_wer(hyps, refs):
    total_words = total_errors = 0
    for hyp, ref in zip(hyps, refs):
        r = ref.lower().split()
        h = hyp.lower().split()
        total_words += len(r)
        total_errors += _edit_distance(r, h)
    return total_errors / max(1, total_words)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--lm", required=True)
    p.add_argument("--penalties", type=float, nargs="+",
                   default=[1.0, 1.05, 1.1, 1.15, 1.2, 1.3])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-text-len", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-batches", type=int, default=None,
                   help="Limit eval batches for quick sweep (None = full set)")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.lm, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.eos_token_id = 248046  # <|im_end|>

    # Load train ds for stats, val ds for eval
    train_ds = BCIDataset(args.data_dir, ALL_SESSIONS, split="train",
                          tokenizer=tokenizer, max_text_len=args.max_text_len, augment=False)
    val_ds = BCIDataset(args.data_dir, ALL_SESSIONS, split="test",
                        tokenizer=tokenizer, max_text_len=args.max_text_len, augment=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=bci_collate_fn, num_workers=0)
    print(f"Test set size: {len(val_ds)} examples")

    model = E2EBCIModel.from_pretrained(
        args.lm, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        n_sessions=len(ALL_SESSIONS),
    )
    model.build_per_session_norm(train_ds.get_session_stats_for_model())

    ckpt_file = os.path.join(args.ckpt, "checkpoint.pt")
    print(f"Loading checkpoint {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()

    # Monkey-patch the rep_penalty per sweep — the model's generate() hardcodes 1.2.
    # We override by patching it at runtime.
    import e2e.model as model_module
    original_generate = model.generate

    results = {}
    for pen in args.penalties:
        print(f"\n=== Sweep: repetition_penalty = {pen} ===")

        # Build a wrapped generate that uses this penalty
        @torch.inference_mode()
        def patched_generate(ecog, ecog_lengths, tokenizer_, max_new_tokens=64,
                              num_beams=1, session_ids=None, _pen=pen):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ecog_embs, ecog_len = model._encode_ecog(ecog, ecog_lengths, session_ids)
                B, T_prime, _ = ecog_embs.shape
                ecog_attn = torch.arange(T_prime, device=ecog.device).unsqueeze(0) < \
                            (ecog_len.unsqueeze(1) if ecog_len is not None
                             else torch.full((B, 1), T_prime, device=ecog.device))
                seed = model._no_think_seed(tokenizer_, ecog.device)
                seed_emb = model._text_embeddings(seed.unsqueeze(0).expand(B, -1))
                inputs_embeds = torch.cat([ecog_embs, seed_emb], dim=1)
                attention_mask = torch.cat(
                    [ecog_attn.long(),
                     torch.ones(B, seed.shape[0], dtype=torch.long, device=ecog.device)],
                    dim=1,
                )
                im_end_id = tokenizer_.convert_tokens_to_ids("<|im_end|>")
                eos_id = im_end_id if im_end_id != tokenizer_.unk_token_id else tokenizer_.eos_token_id
                generated = model.llm.generate(
                    inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens, num_beams=num_beams,
                    eos_token_id=eos_id, pad_token_id=eos_id, do_sample=False,
                    repetition_penalty=_pen,
                )
            texts = tokenizer_.batch_decode(generated, skip_special_tokens=True)
            return [t.strip() for t in texts]

        all_hyps, all_refs = [], []
        for i, batch in enumerate(val_loader):
            if args.max_batches is not None and i >= args.max_batches:
                break
            ecog = batch["ecog"].to(device)
            ecog_len = batch["ecog_lengths"].to(device)
            session_ids = batch["session_idx"].to(device)
            texts = patched_generate(
                ecog, ecog_len, tokenizer,
                max_new_tokens=args.max_new_tokens, num_beams=1,
                session_ids=session_ids,
            )
            all_hyps.extend(texts)
            all_refs.extend(batch["texts"])
            if (i + 1) % 20 == 0:
                wer_so_far = compute_wer(all_hyps, all_refs)
                print(f"  batch {i+1}/{len(val_loader)}  WER so far: {wer_so_far:.4f}")

        wer = compute_wer(all_hyps, all_refs)
        print(f"  rep_penalty={pen}: WER = {wer:.4f}")
        results[str(pen)] = {
            "wer": wer,
            "n_examples": len(all_hyps),
        }

    print(f"\n=== Summary ===")
    best_pen = min(results.keys(), key=lambda k: results[k]["wer"])
    for pen, r in sorted(results.items(), key=lambda kv: float(kv[0])):
        marker = " <-- BEST" if pen == best_pen else ""
        print(f"  rep_penalty={pen}: WER={r['wer']:.4f}{marker}")

    with open(args.output, "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "results": results,
            "best": best_pen,
            "best_wer": results[best_pen]["wer"],
        }, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
