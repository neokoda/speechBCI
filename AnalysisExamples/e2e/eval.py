#!/usr/bin/env python3
"""Evaluate a trained E2E BCI model checkpoint.

Computes WER and CER on the test split and optionally writes per-utterance
predictions to a JSON file for error analysis.

Usage (LLaVA / Qwen model — v4/v5):
    python AnalysisExamples/e2e/eval.py \\
        --data-dir data/derived/tfRecords \\
        --ckpt experiments/e2e_v5/best \\
        --lm Qwen/Qwen3.5-0.8B-Base \\
        --output experiments/e2e_v5/eval_results.json \\
        --beam 1

Usage (Whisper cross-attention model — v6):
    python AnalysisExamples/e2e/eval.py \\
        --data-dir data/derived/tfRecords \\
        --ckpt experiments/e2e_v6/best \\
        --model-type whisper \\
        --whisper-model openai/whisper-medium.en \\
        --output experiments/e2e_v6/eval_results.json \\
        --beam 1
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
from e2e.dataset import BCIDataset, bci_collate_fn

ALL_SESSIONS = [
    "t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17", "t12.2022.05.19",
    "t12.2022.05.24", "t12.2022.05.26", "t12.2022.06.02", "t12.2022.06.07",
    "t12.2022.06.14", "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.23",
    "t12.2022.06.28", "t12.2022.07.05", "t12.2022.07.14", "t12.2022.07.21",
    "t12.2022.07.27", "t12.2022.07.29", "t12.2022.08.02", "t12.2022.08.11",
    "t12.2022.08.13", "t12.2022.08.18", "t12.2022.08.23", "t12.2022.08.25",
]

# Canonical 3-schema slices — see AnalysisExamples/recover_gru_24sess.py:145-169
WILLETT_19 = [
    "t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17", "t12.2022.05.19",
    "t12.2022.05.24", "t12.2022.05.26", "t12.2022.06.02", "t12.2022.06.07",
    "t12.2022.06.14", "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.28",
    "t12.2022.07.05", "t12.2022.07.14", "t12.2022.07.21", "t12.2022.07.27",
    "t12.2022.08.02", "t12.2022.08.11", "t12.2022.08.13",
]
WILLETT_4_18 = WILLETT_19[4:19]  # 15 sessions

SLICES = {
    "all_24":       set(range(len(ALL_SESSIONS))),
    "willett_19":   {ALL_SESSIONS.index(s) for s in WILLETT_19},
    "willett_4_18": {ALL_SESSIONS.index(s) for s in WILLETT_4_18},
}


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


def compute_wer(hyps: list[str], refs: list[str], sess_idx: list = None) -> tuple[float, list]:
    """Corpus-level (micro-averaged) WER = total_errors / total_words.
    If sess_idx provided, each detail row carries the session index."""
    total_words = total_errors = 0
    details = []
    for k, (hyp, ref) in enumerate(zip(hyps, refs)):
        r = ref.lower().split()
        h = hyp.lower().split()
        err = _edit_distance(r, h)
        total_words  += len(r)
        total_errors += err
        row = {"ref": ref, "hyp": hyp,
               "wer": err / max(1, len(r)), "errors": err, "words": len(r)}
        if sess_idx is not None:
            row["session_idx"] = int(sess_idx[k])
        details.append(row)
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


def slice_metrics(hyps, refs, sess_idx):
    """Compute corpus-level WER/CER per slice (all_24, willett_19, willett_4_18)."""
    out = {}
    for name, idx_set in SLICES.items():
        sub_h, sub_r = [], []
        for h, r, s in zip(hyps, refs, sess_idx):
            if int(s) in idx_set:
                sub_h.append(h); sub_r.append(r)
        if not sub_h:
            out[name] = {"wer": None, "cer": None, "n": 0}
            continue
        wer, _ = compute_wer(sub_h, sub_r)
        cer = compute_cer(sub_h, sub_r)
        out[name] = {"wer": wer, "cer": cer, "n": len(sub_h)}
    return out


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Tokenizer & model ─────────────────────────────────────────────────
    if args.model_type == "whisper":
        from e2e.whisper_model import WhisperBCIModel

        tokenizer = AutoTokenizer.from_pretrained(args.whisper_model, use_fast=True)

        sessions = ALL_SESSIONS
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

        print(f"Loading WhisperBCIModel from {args.ckpt}")
        model = WhisperBCIModel(
            whisper_name=args.whisper_model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            cross_attn_only=args.cross_attn_only,
            n_sessions=len(sessions),
        )
        model.build_per_session_norm(session_stats)

    elif args.model_type == "canary":
        from e2e.canary_model import CanaryBCIModel

        qwen_name = args.lm or "Qwen/Qwen3-1.7B"
        tokenizer = AutoTokenizer.from_pretrained(qwen_name, trust_remote_code=True, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.eos_token_id = 248046  # <|im_end|> for Qwen3

        sessions = ALL_SESSIONS
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

        print(f"Loading CanaryBCIModel from {args.ckpt}")
        model = CanaryBCIModel(
            qwen_name=qwen_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            n_sessions=len(sessions),
        )
        model.build_per_session_norm(session_stats)

    else:
        from e2e.model import E2EBCIModel

        tokenizer = AutoTokenizer.from_pretrained(
            args.lm, trust_remote_code=True, use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.eos_token_id = 248046  # <|im_end|> for Qwen3.5

        sessions = ALL_SESSIONS
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

        print(f"Loading E2EBCIModel from {args.ckpt}")
        model = E2EBCIModel.from_pretrained(
            args.lm,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            n_sessions=len(sessions),
        )
        model.build_per_session_norm(session_stats)

    # ── Load checkpoint weights ────────────────────────────────────────────
    ckpt_file = os.path.join(args.ckpt, "checkpoint.pt")
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint at step {ckpt.get('step', '?')}")

    # ── Decode all batches ─────────────────────────────────────────────────
    all_hyps, all_refs, all_sess = [], [], []
    for i, batch in enumerate(loader):
        ecog        = batch["ecog"].to(device)
        ecog_len    = batch["ecog_lengths"].to(device)
        session_ids = batch["session_idx"].to(device)

        texts = model.generate(
            ecog, ecog_len, tokenizer,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.beam,
            session_ids=session_ids,
        )
        all_hyps.extend(texts)
        all_refs.extend(batch["texts"])
        all_sess.extend(session_ids.detach().cpu().tolist())

        if (i + 1) % 20 == 0:
            wer_so_far, _ = compute_wer(all_hyps, all_refs)
            print(f"  batch {i+1}/{len(loader)}  WER so far: {wer_so_far:.4f}")

    # ── Metrics ───────────────────────────────────────────────────────────
    wer, details = compute_wer(all_hyps, all_refs, sess_idx=all_sess)
    cer = compute_cer(all_hyps, all_refs)
    slices = slice_metrics(all_hyps, all_refs, all_sess)

    print(f"\n{'='*60}")
    print(f"  WER (all_24): {wer:.4f}   CER (all_24): {cer:.4f}")
    for name in ("all_24", "willett_19", "willett_4_18"):
        s = slices[name]
        if s["wer"] is None:
            print(f"  {name:>14s}: n={s['n']}  (empty subset)")
        else:
            print(f"  {name:>14s}: WER={s['wer']:.4f}  CER={s['cer']:.4f}  n={s['n']}")
    print(f"  (baseline two-stage: WER=0.1895 / CER=0.1362)")
    print(f"{'='*60}\n")

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
            "checkpoint":   args.ckpt,
            "model_type":   args.model_type,
            "beam":         args.beam,
            "wer":          wer,
            "cer":          cer,
            "slices":       slices,
            "n_examples":   len(all_hyps),
            "baseline_wer": 0.1895,
            "baseline_cer": 0.1362,
            "details":      details,
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
    p.add_argument("--data-dir",       required=True)
    p.add_argument("--ckpt",           required=True,
                   help="Checkpoint directory (contains checkpoint.pt)")
    p.add_argument("--model-type",     default="llava", choices=["llava", "whisper", "canary"],
                   help="llava=E2EBCIModel (Qwen), whisper=WhisperBCIModel, canary=CanaryBCIModel")
    # LLaVA model args
    p.add_argument("--lm",             default=None,
                   help="HuggingFace LM id (required for --model-type llava)")
    # Whisper model args
    p.add_argument("--whisper-model",  default="openai/whisper-medium.en")
    p.add_argument("--output",         default=None)
    p.add_argument("--batch-size",     type=int, default=4)
    p.add_argument("--beam",           type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-text-len",   type=int, default=64)
    p.add_argument("--lora-r",         type=int, default=16,
                   help="default 16, use 128 for canary")
    p.add_argument("--lora-alpha",     type=int, default=32,
                   help="default 32, use 256 for canary")
    p.add_argument("--cross-attn-only", action="store_true")
    p.add_argument("--hf-token",       default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.model_type not in ("llava", "canary") and args.lm is None:
        raise ValueError("--lm is required for --model-type llava")
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
    evaluate(args)
