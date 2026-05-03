#!/usr/bin/env python3
"""Evaluate a trained E2E BCI model checkpoint.

Computes WER and CER on the test split and optionally writes per-utterance
predictions to a JSON file for error analysis.

Usage:
    python AnalysisExamples/e2e/eval.py \\
        --data-dir data/derived/tfRecords \\
        --ckpt experiments/e2e_0.8b/best \\
        --lm Qwen/Qwen3.5-0.8B-Base \\
        --output experiments/e2e_0.8b/eval_results.json \\
        --beam 1

    # Compare multiple checkpoints quickly
    for ckpt in experiments/e2e_0.8b/best experiments/e2e_2b/best; do
        python AnalysisExamples/e2e/eval.py --ckpt $ckpt --data-dir ...
    done
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _edit_distance(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for ai in a:
        nd = [d[0] + 1]
        for j, bi in enumerate(b):
            nd.append(min(d[j + 1] + 1, nd[j] + 1, d[j] + (0 if ai == bi else 1)))
        d = nd
    return d[len(b)]


def compute_wer(hyps: list[str], refs: list[str]) -> tuple[float, dict]:
    total_words = total_errors = 0
    details = []
    for hyp, ref in zip(hyps, refs):
        r = ref.lower().split()
        h = hyp.lower().split()
        err = _edit_distance(r, h)
        total_words  += len(r)
        total_errors += err
        details.append({"ref": ref, "hyp": hyp,
                        "wer": err / max(1, len(r)), "errors": err, "words": len(r)})
    wer = total_errors / max(1, total_words)
    return wer, details


def compute_cer(hyps: list[str], refs: list[str]) -> float:
    total_chars = total_errors = 0
    for hyp, ref in zip(hyps, refs):
        r = list(ref.lower().replace(" ", ""))
        h = list(hyp.lower().replace(" ", ""))
        total_chars  += len(r)
        total_errors += _edit_distance(r, h)
    return total_errors / max(1, total_chars)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.lm, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Qwen3.5 tokenizer bug: bos_token_id=None, eos_token_id=248044 (<|endoftext|>).
    # Use 248046 (<|im_end|>) as the real text EOS and block 248044 during generation.
    tokenizer.eos_token_id = 248046

    # ── Dataset ───────────────────────────────────────────────────────────
    sessions = ALL_SESSIONS
    # Load train dataset to get per-session normalization stats
    train_ds = BCIDataset(
        args.data_dir, sessions, split="train",
        tokenizer=tokenizer, max_text_len=args.max_text_len, augment=False,
    )
    session_stats = train_ds.get_session_stats_for_model()

    ds = BCIDataset(
        args.data_dir, sessions, split="test",
        tokenizer=tokenizer, max_text_len=args.max_text_len, augment=False,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=bci_collate_fn, num_workers=0,
    )
    print(f"Test examples: {len(ds)}")

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"Loading model from {args.ckpt}")
    model = E2EBCIModel.from_pretrained(
        args.lm,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        n_sessions=len(sessions),
    )
    model.build_per_session_norm(session_stats)

    ckpt_file = os.path.join(args.ckpt, "checkpoint.pt")
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint at step {ckpt.get('step', '?')}")

    # ── Decode all batches ─────────────────────────────────────────────────
    all_hyps, all_refs = [], []
    for i, batch in enumerate(loader):
        ecog     = batch["ecog"].to(device)
        ecog_len = batch["ecog_lengths"].to(device)
        session_ids = batch["session_idx"].to(device)

        texts = model.generate(
            ecog, ecog_len, tokenizer,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.beam,
            session_ids=session_ids,
        )
        all_hyps.extend(texts)
        all_refs.extend(batch["texts"])

        if (i + 1) % 20 == 0:
            wer_so_far, _ = compute_wer(all_hyps, all_refs)
            print(f"  batch {i+1}/{len(loader)}  WER so far: {wer_so_far:.4f}")

    # ── Metrics ───────────────────────────────────────────────────────────
    wer, details = compute_wer(all_hyps, all_refs)
    cer = compute_cer(all_hyps, all_refs)

    print(f"\n{'='*60}")
    print(f"  WER: {wer:.4f}   CER: {cer:.4f}")
    print(f"  (baseline two-stage: WER=0.1895 / CER=0.1362)")
    print(f"{'='*60}\n")

    # ── Sample outputs ────────────────────────────────────────────────────
    print("Sample predictions:")
    for d in details[:5]:
        print(f"  REF: {d['ref']}")
        print(f"  HYP: {d['hyp']}")
        print(f"  WER: {d['wer']:.3f}")
        print()

    # ── Save results ──────────────────────────────────────────────────────
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        results = {
            "checkpoint": args.ckpt,
            "lm":         args.lm,
            "beam":       args.beam,
            "wer":        wer,
            "cer":        cer,
            "n_examples": len(all_hyps),
            "baseline_wer": 0.1895,
            "baseline_cer": 0.1362,
            "details":    details,
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

    return wer, cer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate E2E BCI model")
    p.add_argument("--data-dir",      required=True)
    p.add_argument("--ckpt",          required=True,
                   help="Checkpoint directory (contains checkpoint.pt)")
    p.add_argument("--lm",            required=True,
                   help="Same HF model id used during training")
    p.add_argument("--output",        default=None,
                   help="Path to write JSON results (optional)")
    p.add_argument("--batch-size",    type=int, default=4)
    p.add_argument("--beam",          type=int, default=1,
                   help="Beam width (1=greedy)")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-text-len",  type=int, default=64)
    p.add_argument("--lora-r",        type=int, default=16)
    p.add_argument("--lora-alpha",    type=int, default=32)
    p.add_argument("--hf-token",      default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
    evaluate(args)
