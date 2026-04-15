#!/usr/bin/env python3
"""
Download + tokenize OpenWebText for LLaMA-2 QLoRA fine-tuning.

Uses `Skylion007/openwebtext` (most-downloaded OWT on HF, ~8M docs, ~40 GB).
Seto's thesis didn't specify OWT1 vs the Pile's OWT2; this is the canonical
community rebuild and the de-facto "OWT2" in most recent papers.

Output: Arrow shards of packed 1024-token sequences under --output-dir.
"""
import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer


def pack_sequences(ids_stream, seq_len, eos_id):
    buf = []
    for ids in ids_stream:
        buf.extend(ids)
        buf.append(eos_id)
        while len(buf) >= seq_len:
            yield buf[:seq_len]
            buf = buf[seq_len:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default='data/owt2_hf')
    ap.add_argument('--model-id', default='meta-llama/Llama-2-7b-hf')
    ap.add_argument('--seq-len', type=int, default=1024)
    ap.add_argument('--max-tokens', type=int, default=500_000_000,
                    help='cap total tokens for one-epoch training; ~500M is ~3-6h on 4090')
    ap.add_argument('--val-tokens', type=int, default=2_000_000)
    ap.add_argument('--hf-token', default=None)
    ap.add_argument('--cache-dir', default='/root/.cache/huggingface')
    ap.add_argument('--dataset-id', default='Skylion007/openwebtext')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tok_kwargs = {'cache_dir': args.cache_dir}
    if args.hf_token:
        tok_kwargs['token'] = args.hf_token
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **tok_kwargs)
    eos_id = tokenizer.eos_token_id

    print(f'Streaming {args.dataset_id} ...', flush=True)
    ds = load_dataset(args.dataset_id, split='train', streaming=True,
                      cache_dir=args.cache_dir, trust_remote_code=True)

    def ids_iter():
        for i, ex in enumerate(ds):
            text = ex['text']
            if not text or len(text) < 64:
                continue
            yield tokenizer(text, add_special_tokens=False)['input_ids']
            if i % 10000 == 0 and i > 0:
                print(f'  tokenized {i} docs', flush=True)

    target = args.max_tokens + args.val_tokens
    packed = []
    tot = 0
    for seq in pack_sequences(ids_iter(), args.seq_len, eos_id):
        packed.append(seq)
        tot += len(seq)
        if len(packed) % 1000 == 0:
            print(f'  packed {len(packed)} seqs ({tot/1e6:.1f}M tokens)', flush=True)
        if tot >= target:
            break

    n_val = args.val_tokens // args.seq_len
    val = packed[:n_val]
    train = packed[n_val:]
    print(f'Saving {len(train)} train / {len(val)} val sequences', flush=True)

    from datasets import Dataset
    Dataset.from_dict({'input_ids': train}).save_to_disk(os.path.join(args.output_dir, 'train'))
    Dataset.from_dict({'input_ids': val}).save_to_disk(os.path.join(args.output_dir, 'val'))
    print('Done.')


if __name__ == '__main__':
    main()
