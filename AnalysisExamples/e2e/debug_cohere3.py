"""Deeper Cohere debug: determinism + tied-weight + hyp samples."""
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from e2e.cohere_model import CohereBCIModel, COHERE_REPO_DEFAULT
from e2e.dataset import BCIDataset, bci_collate_fn
from e2e.train_cohere import validate, ALL_SESSIONS


def main():
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(COHERE_REPO_DEFAULT, trust_remote_code=True)
    train_ds = BCIDataset("data/derived/tfRecords", ALL_SESSIONS, split="train",
                          tokenizer=tokenizer, max_text_len=64, augment=False)
    session_stats = train_ds.get_session_stats_for_model()
    test_ds = BCIDataset("data/derived/tfRecords", ALL_SESSIONS, split="test",
                         tokenizer=tokenizer, max_text_len=64, augment=False)
    loader = DataLoader(test_ds, batch_size=8, shuffle=False,
                        collate_fn=bci_collate_fn, num_workers=0)

    m = CohereBCIModel(n_sessions=24, lora_r=16, lora_alpha=32, lora_dropout=0.1)
    m.build_per_session_norm(session_stats)
    ckpt = torch.load("experiments/e2e_cohere/best/checkpoint.pt",
                      map_location="cpu", weights_only=False)
    m.load_state_dict(ckpt["model"], strict=False)
    m.to(device)

    # Tied weight integrity
    base = m.cohere.base_model.model
    tied = base.transf_decoder._embedding.token_embedding.weight is base.log_softmax.mlp.layer0.weight
    print(f"tied weight still ID-equal post-load: {tied}")
    print(f"  token_emb norm: {base.transf_decoder._embedding.token_embedding.weight.norm().item():.4f}")
    print(f"  lm_head   norm: {base.log_softmax.mlp.layer0.weight.norm().item():.4f}")

    # Active adapter
    print(f"\nactive_adapter: {m.cohere.active_adapter}")

    # Run validate twice
    vl1, vw1 = validate(m, loader, tokenizer, device, max_batches=10)
    vl2, vw2 = validate(m, loader, tokenizer, device, max_batches=10)
    print(f"\nvalidate #1: loss={vl1:.4f}  WER={vw1:.4f}")
    print(f"validate #2: loss={vl2:.4f}  WER={vw2:.4f}")

    # Also run forward-only over first 3 batches and print logits stats
    print("\n--- forward-only inspection ---")
    m.eval()
    for i, batch in enumerate(loader):
        if i >= 1: break
        ecog = batch["ecog"].to(device)
        ecog_len = batch["ecog_lengths"].to(device)
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        sess = batch["session_idx"].to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ecog_mem, enc_len = m._encode_ecog(ecog, ecog_len, sess)
            print(f"ECoG memory stats: shape={ecog_mem.shape}  mean={ecog_mem.float().mean().item():.4f}  std={ecog_mem.float().std().item():.4f}")
            from transformers.modeling_outputs import BaseModelOutput
            cross_mask = m._cross_attention_mask(ecog_mem, enc_len)
            out = m.cohere(
                encoder_outputs=BaseModelOutput(last_hidden_state=ecog_mem),
                cross_attention_mask=cross_mask,
                decoder_input_ids=ids,
                decoder_attention_mask=attn,
                labels=labels,
            )
            print(f"loss this batch: {out.loss.item():.4f}")
            print(f"logits shape: {out.logits.shape}  argmax sample: {out.logits[0,5].argmax().item()}  refers tok: {labels[0,5].item()}")


if __name__ == "__main__":
    main()
