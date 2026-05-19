"""Side-by-side reproduction of Cohere training-time validate vs eval.py paths.

Goal: explain the 0.60 partial-val vs 5.68 full-eval gap.

For each of three configurations we run the first N_BATCHES batches of the test
loader through the trained checkpoint and print per-utterance hyp/ref + WER:

  A. eval.py path        — generate() outside any autocast (fp32 params, fp32 fwd)
  B. validate() path     — loss forward in autocast(bf16) then generate() outside
  C. autocasted generate — generate() wrapped in autocast(bf16)

If (A) gives WER ~5 and (C) gives WER ~0.6, the bug is dtype: training's bf16
generate happened to land in a different decoder regime than fp32 generate.

If all three give ~0.6, the bug is somewhere in eval.py (likely the dataset/loader
construction or the model state at load time).

If all three give ~5, the original 0.60 number was misreported and the model
really is broken at this checkpoint.
"""
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from e2e.cohere_model import CohereBCIModel, COHERE_REPO_DEFAULT
from e2e.dataset import BCIDataset, bci_collate_fn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ALL_SESSIONS = [
    "t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17", "t12.2022.05.19",
    "t12.2022.05.24", "t12.2022.05.26", "t12.2022.06.02", "t12.2022.06.07",
    "t12.2022.06.14", "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.23",
    "t12.2022.06.28", "t12.2022.07.05", "t12.2022.07.14", "t12.2022.07.21",
    "t12.2022.07.27", "t12.2022.07.29", "t12.2022.08.02", "t12.2022.08.11",
    "t12.2022.08.13", "t12.2022.08.18", "t12.2022.08.23", "t12.2022.08.25",
]


def compute_wer(hyps, refs):
    total_words = total_errors = 0
    for hyp, ref in zip(hyps, refs):
        r = ref.lower().split()
        h = hyp.lower().split()
        total_words += len(r)
        d = list(range(len(h) + 1))
        for rw in r:
            nd = [d[0] + 1]
            for hi, hw in enumerate(h):
                nd.append(min(d[hi + 1] + 1, nd[hi] + 1, d[hi] + (0 if rw == hw else 1)))
            d = nd
        total_errors += d[len(h)]
    return total_errors / max(1, total_words)


def run_path(label, model, loader, tokenizer, device, n_batches, do_fwd_in_bf16, gen_in_bf16):
    print(f"\n========== {label} ==========")
    was_training = model.training
    model.eval()
    hyps, refs = [], []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        ecog        = batch["ecog"].to(device)
        ecog_len    = batch["ecog_lengths"].to(device)
        ids         = batch["input_ids"].to(device)
        attn        = batch["attention_mask"].to(device)
        labels      = batch["labels"].to(device)
        session_ids = batch["session_idx"].to(device)

        if do_fwd_in_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
                _ = model(ecog, ecog_len, ids, attn, labels, session_ids)
        if gen_in_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
                texts = model.generate(ecog, ecog_len, tokenizer, max_new_tokens=64,
                                       num_beams=1, session_ids=session_ids)
        else:
            with torch.no_grad():
                texts = model.generate(ecog, ecog_len, tokenizer, max_new_tokens=64,
                                       num_beams=1, session_ids=session_ids)
        hyps.extend(texts)
        refs.extend(batch["texts"])
    model.train(was_training)

    wer = compute_wer(hyps, refs)
    print(f"  WER over {len(hyps)} utts = {wer:.4f}")
    for j in range(min(4, len(hyps))):
        print(f"  [{j}] REF: {refs[j]!r}")
        print(f"      HYP: {hyps[j]!r}")
    return wer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/derived/tfRecords")
    ap.add_argument("--ckpt", default="experiments/e2e_cohere/best")
    ap.add_argument("--cohere-repo", default=COHERE_REPO_DEFAULT)
    ap.add_argument("--n-batches", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.cohere_repo, trust_remote_code=True)
    train_ds = BCIDataset(args.data_dir, ALL_SESSIONS, split="train",
                          tokenizer=tokenizer, max_text_len=64, augment=False)
    session_stats = train_ds.get_session_stats_for_model()
    test_ds = BCIDataset(args.data_dir, ALL_SESSIONS, split="test",
                         tokenizer=tokenizer, max_text_len=64, augment=False)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=bci_collate_fn, num_workers=0)

    print(f"Loading CohereBCIModel from {args.ckpt}")
    model = CohereBCIModel(cohere_repo=args.cohere_repo, lora_r=16, lora_alpha=32,
                           n_sessions=len(ALL_SESSIONS))
    model.build_per_session_norm(session_stats)
    ckpt_file = os.path.join(args.ckpt, "checkpoint.pt")
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    print(f"Loaded step {ckpt.get('step')}  best_wer {ckpt.get('best_wer')}")

    wer_A = run_path("A: generate() in fp32 (eval.py path)",
                     model, loader, tokenizer, device, args.n_batches,
                     do_fwd_in_bf16=False, gen_in_bf16=False)
    wer_B = run_path("B: bf16 forward then fp32 generate (train validate path)",
                     model, loader, tokenizer, device, args.n_batches,
                     do_fwd_in_bf16=True, gen_in_bf16=False)
    wer_C = run_path("C: generate() in bf16 autocast",
                     model, loader, tokenizer, device, args.n_batches,
                     do_fwd_in_bf16=False, gen_in_bf16=True)

    print("\n==== SUMMARY ====")
    print(f"  A (fp32 generate)             WER={wer_A:.4f}")
    print(f"  B (bf16 fwd + fp32 generate)  WER={wer_B:.4f}")
    print(f"  C (bf16 generate)             WER={wer_C:.4f}")


if __name__ == "__main__":
    main()
