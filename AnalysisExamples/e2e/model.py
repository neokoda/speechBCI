"""E2E BCI Model — Architecture A (LLaVA-style decoder-only).

Pipeline:
    ECoG (B, T, 256)
        ↓  ConformerEncoder      → (B, T', d_model)
        ↓  MLPProjector          → (B, T', llm_dim)
        ↓  concat with text embs → (B, T'+text_len, llm_dim)
        ↓  Decoder-only LLM      → logits on text positions only

Usage:
    model = E2EBCIModel.from_pretrained("Qwen/Qwen3.5-2B-Base")
    loss  = model(ecog, ecog_lengths, input_ids, attention_mask, labels)
    text  = model.generate(ecog, ecog_lengths, tokenizer, max_new_tokens=64)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType

from .conformer_pt import ConformerEncoder


# ---------------------------------------------------------------------------
# MLP projector (ECoG d_model → LLM embedding dim)
# ---------------------------------------------------------------------------

class MLPProjector(nn.Module):
    """2-layer MLP with GELU.  Zero-init on the output layer for stable
    early training (follows LLaVA 1.5 init strategy).
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class E2EBCIModel(nn.Module):
    """End-to-end BCI decoder: Conformer + MLP projector + decoder-only LLM.

    The LLM sees:
        [ecog_token_0, ..., ecog_token_N, BOS, tok_0, ..., tok_M]

    Cross-entropy loss is computed only on the text positions (LLM auto-
    regressively generates the transcription given the ECoG prefix).
    """

    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

    def __init__(
        self,
        llm_name_or_path: str,
        # Conformer
        n_input: int          = 256,
        d_model: int          = 512,
        nhead: int            = 8,
        num_layers: int       = 4,
        d_ff: int             = 2048,
        conv_kernel_size: int = 31,
        stem_kernel: int      = 32,
        stem_stride: int      = 4,
        dropout: float        = 0.1,
        spatial_attention: bool = True,
        spec_augment: bool    = False,
        # LoRA
        lora_r: int           = 16,
        lora_alpha: int       = 32,
        lora_dropout: float   = 0.05,
        # Misc
        freeze_llm: bool      = False,   # set True for Phase-1 warmup
        freeze_encoder: bool  = False,   # set True for Phase-1 warmup
        torch_dtype           = torch.bfloat16,
    ):
        super().__init__()

        # ── Conformer encoder ─────────────────────────────────────────────
        self.encoder = ConformerEncoder(
            n_input=n_input,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            d_ff=d_ff,
            conv_kernel_size=conv_kernel_size,
            stem_kernel=stem_kernel,
            stem_stride=stem_stride,
            dropout=dropout,
            spatial_attention=spatial_attention,
            spec_augment=spec_augment,
        )

        # ── Load LLM ──────────────────────────────────────────────────────
        llm_config = AutoConfig.from_pretrained(llm_name_or_path, trust_remote_code=True)
        # VL models (e.g. Qwen3.5) wrap text config inside .text_config
        self.llm_dim = getattr(llm_config, "hidden_size", None) or llm_config.text_config.hidden_size

        base_llm = AutoModelForCausalLM.from_pretrained(
            llm_name_or_path,
            dtype=torch_dtype,
            trust_remote_code=True,
        )

        # Apply LoRA
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=self.LORA_TARGET_MODULES,
            bias="none",
        )
        self.llm = get_peft_model(base_llm, lora_cfg)
        self.llm.enable_input_require_grads()
        self.llm.gradient_checkpointing_enable()

        # ── MLP projector ─────────────────────────────────────────────────
        self.projector = MLPProjector(d_model, self.llm_dim)

        # ── Optional freezing (Phase 1) ───────────────────────────────────
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad_(False)
            # But always keep LoRA trainable when not in phase-1 of projector-only
            # (freeze_llm=True only during the very short projector warmup)

    # ------------------------------------------------------------------
    # Forward (training / teacher-forcing)
    # ------------------------------------------------------------------

    def forward(
        self,
        ecog: torch.Tensor,              # (B, T, 256)
        ecog_lengths: torch.Tensor,      # (B,)
        input_ids: torch.Tensor,         # (B, L_text) — [BOS tok_0 … tok_M]
        attention_mask: torch.Tensor,    # (B, L_text)
        labels: torch.Tensor,            # (B, L_text) — [-100 … BOS tok_0 … EOS]
    ) -> torch.Tensor:
        """Return scalar CE loss (text positions only)."""

        ecog_embs, ecog_len = self._encode_ecog(ecog, ecog_lengths)
        # ecog_embs: (B, T', llm_dim)

        # Build full input embeddings: [ECoG prefix | text tokens]
        text_embs = self._text_embeddings(input_ids)  # (B, L_text, llm_dim)
        inputs_embeds, full_attention_mask, full_labels = self._concat_inputs(
            ecog_embs, ecog_len, text_embs, attention_mask, labels
        )

        # Run LLM without labels to get raw logits, then compute CE only on
        # text positions — avoids materialising a (B, T'+L_text, vocab) tensor
        # for the loss, cutting peak memory roughly in half.
        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
        )
        # out.logits: (B, T'+L_text, V).  Text labels start at position T'.
        T_prime = ecog_embs.shape[1]
        # Shift: position i predicts label i+1
        text_logits = out.logits[:, T_prime - 1 : -1, :]   # (B, L_text, V)
        text_labels = labels                                  # (B, L_text)
        loss = F.cross_entropy(
            text_logits.reshape(-1, text_logits.size(-1)),
            text_labels.reshape(-1),
            ignore_index=-100,
        )
        return loss

    # ------------------------------------------------------------------
    # Generation (inference)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        ecog: torch.Tensor,
        ecog_lengths: torch.Tensor,
        tokenizer,
        max_new_tokens: int = 64,
        num_beams: int      = 1,
    ) -> list[str]:
        """Decode ECoG → text strings."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ecog_embs, ecog_len = self._encode_ecog(ecog, ecog_lengths)
        B, T_prime, _ = ecog_embs.shape

        # Attention mask for ECoG tokens
        ecog_attn = torch.arange(T_prime, device=ecog.device).unsqueeze(0) < \
                    (ecog_len.unsqueeze(1) if ecog_len is not None
                     else torch.full((B, 1), T_prime, device=ecog.device))

        # BOS token to kick off generation
        bos_id  = tokenizer.bos_token_id or tokenizer.eos_token_id
        bos_emb = self._text_embeddings(
            torch.full((B, 1), bos_id, dtype=torch.long, device=ecog.device)
        )

        inputs_embeds  = torch.cat([ecog_embs, bos_emb], dim=1)
        attention_mask = torch.cat(
            [ecog_attn.long(), torch.ones(B, 1, dtype=torch.long, device=ecog.device)],
            dim=1
        )

        generated = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            do_sample=False,
        )
        # generated contains only the new tokens (past the prompt)
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [t.strip() for t in texts]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_ecog(self, ecog, ecog_lengths):
        enc_out, enc_len = self.encoder(ecog.float(), ecog_lengths)  # (B,T',d_model)
        return self.projector(enc_out.to(self.projector.net[0].weight.dtype)), enc_len

    def _text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        base = self.llm.get_base_model()
        # Qwen3/LLaMA: model.embed_tokens; Gemma: model.embed_tokens; fallback via get_input_embeddings()
        if hasattr(base, "model") and hasattr(base.model, "embed_tokens"):
            emb = base.model.embed_tokens
        else:
            emb = base.get_input_embeddings()
        return emb(input_ids)

    def _concat_inputs(self, ecog_embs, ecog_len, text_embs, text_attn, labels):
        B, T_prime, D = ecog_embs.shape
        _, L_text, _  = text_embs.shape

        # ECoG attention mask derived from valid lengths
        if ecog_len is not None:
            ecog_attn = (
                torch.arange(T_prime, device=ecog_embs.device).unsqueeze(0) <
                ecog_len.unsqueeze(1)
            ).long()
        else:
            ecog_attn = torch.ones(B, T_prime, dtype=torch.long,
                                   device=ecog_embs.device)

        inputs_embeds   = torch.cat([ecog_embs, text_embs], dim=1)
        attention_mask  = torch.cat([ecog_attn, text_attn], dim=1)

        # Labels: mask out ECoG prefix positions with -100
        ecog_ignore = torch.full((B, T_prime), -100, dtype=torch.long,
                                  device=ecog_embs.device)
        full_labels = torch.cat([ecog_ignore, labels], dim=1)

        return inputs_embeds, attention_mask, full_labels

    # ------------------------------------------------------------------
    # Utility: print trainable param count
    # ------------------------------------------------------------------

    def print_trainable_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")

    # ------------------------------------------------------------------
    # Convenience constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, llm_name_or_path: str, **kwargs) -> "E2EBCIModel":
        return cls(llm_name_or_path=llm_name_or_path, **kwargs)
