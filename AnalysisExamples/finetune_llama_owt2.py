#!/usr/bin/env python3
"""
QLoRA fine-tune LLaMA-2 7B on OWT2 for BSSF rescoring.

Consumes Arrow shards from scripts/prepare_owt2.py. Saves LoRA adapters to
--output-dir, loadable by rescore_nbest.py via --lora-dir.

Target: close the gap from 0.1895 (stock LLaMA-2 BSSF) toward Seto's 0.169.

VRAM budget on RTX 4090 (24 GB): 4-bit NF4 base + LoRA r=16 + grad ckpt
at seq_len 1024 ≈ 14–17 GB. Do not raise seq_len without checking.
"""
import argparse
import os
import sys

# Re-exec with venv CUDA libs on LD_LIBRARY_PATH (same trick as rescore_nbest.py).
import site as _site
_NV_BASE = os.path.join(_site.getsitepackages()[0], 'nvidia')
if os.path.isdir(_NV_BASE):
    _NV_SUBDIRS = ['cudnn', 'cublas', 'cuda_nvrtc', 'cuda_runtime',
                   'cufft', 'cusolver', 'cusparse', 'nvjitlink']
    _NV_LIBS = ':'.join(
        os.path.join(_NV_BASE, d, 'lib') for d in _NV_SUBDIRS
        if os.path.isdir(os.path.join(_NV_BASE, d, 'lib'))
    )
    _cur = os.environ.get('LD_LIBRARY_PATH', '')
    if _NV_LIBS and _NV_LIBS.split(':')[0] not in _cur:
        _env = os.environ.copy()
        _env['LD_LIBRARY_PATH'] = _NV_LIBS + ':' + _cur
        os.execve(sys.executable, [sys.executable] + sys.argv, _env)

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    DataCollatorForLanguageModeling, EarlyStoppingCallback,
    Trainer, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data/owt2_hf')
    ap.add_argument('--output-dir', default='experiments/llama2_owt2_lora')
    ap.add_argument('--model-id', default='meta-llama/Llama-2-7b-hf')
    ap.add_argument('--lora-r', type=int, default=16)
    ap.add_argument('--lora-alpha', type=int, default=32)
    ap.add_argument('--lora-dropout', type=float, default=0.05)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--grad-accum', type=int, default=1)
    ap.add_argument('--epochs', type=float, default=1.0)
    ap.add_argument('--max-steps', type=int, default=-1,
                    help='override epoch budget; -1 means use --epochs')
    ap.add_argument('--warmup-steps', type=int, default=1000)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--save-steps', type=int, default=500)
    ap.add_argument('--eval-steps', type=int, default=500)
    ap.add_argument('--logging-steps', type=int, default=25)
    ap.add_argument('--early-stopping-patience', type=int, default=3,
                    help='stop if eval_loss fails to improve for this many evals')
    ap.add_argument('--hf-token', default=None)
    ap.add_argument('--cache-dir', default='/root/.cache/huggingface')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tok_kwargs = {'cache_dir': args.cache_dir}
    if args.hf_token:
        tok_kwargs['token'] = args.hf_token
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_kwargs = {'cache_dir': args.cache_dir, 'quantization_config': bnb_cfg,
                    'torch_dtype': torch.bfloat16}
    if args.hf_token:
        model_kwargs['token'] = args.hf_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        bias='none', task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train = load_from_disk(os.path.join(args.data_dir, 'train'))
    val = load_from_disk(os.path.join(args.data_dir, 'val'))
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type='cosine',
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_steps=args.logging_steps,
        save_strategy='steps', save_steps=args.save_steps, save_total_limit=3,
        eval_strategy='steps', eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to='none',
        optim='paged_adamw_8bit',
    )
    trainer = Trainer(
        model=model, args=training_args, processing_class=tokenizer,
        train_dataset=train, eval_dataset=val, data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    resume = any(d.startswith('checkpoint-') for d in os.listdir(args.output_dir))
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    print(f'LoRA adapter saved to {args.output_dir}')


if __name__ == '__main__':
    main()
