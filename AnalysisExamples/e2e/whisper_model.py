"""WhisperBCI Model — Whisper cross-attention E2E BCI decoder.

Pipeline:
    ECoG (B, T, 256)
        ↓  ConformerEncoder      → (B, T', d_model)
        ↓  Linear + LayerNorm    → (B, T', 1024)     [Whisper medium d_model]
        = ECoG memory  ─────────────────────────────────────────────┐
                                                                    │ K, V (Whisper cross-attn)
    Text (teacher-forced at train / autoregressive at inference):  │
        ↓  Whisper embedding    → (B, L, 1024)                    │
        ↓  Causal self-attn     (text → text only, causal)        │
        ↓  Cross-attention      (Q=text, K=V=ECoG memory)  ←──────┘
        ↓  FFN
        × 24 Whisper decoder layers (pre-trained, LoRA'd)
        ↓  LM head              → logits (B, L, 51864)

ECoG is never in the text self-attention window — the only path from ECoG to
text predictions is cross-attention. This eliminates the text-shortcut problem
present in LLaVA-style concatenation models (v4/v5).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperConfig
from transformers.modeling_outputs import BaseModelOutput
from peft import get_peft_model, LoraConfig

from .conformer_pt import ConformerEncoder


class WhisperBCIModel(nn.Module):
    """End-to-end BCI decoder: Conformer + projector + Whisper decoder cross-attention."""

    # Attention projections + FFN. The FFN (`fc1`/`fc2`) carries most of Whisper's
    # speech knowledge (trained on 680k hr of Mel spectrograms). Attention-only LoRA
    # can't adapt those features to ECoG input. v9+ includes FFN targets; older
    # checkpoints (v6, v7) only had attention LoRA, but load_state_dict(..., strict=False)
    # leaves the new fc1/fc2 LoRA at fresh PEFT init (lora_B zero → no contribution at step 0).
    LORA_TARGET_MODULES       = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    LORA_CROSS_ATTN_MODULES   = ["encoder_attn.q_proj", "encoder_attn.k_proj",
                                  "encoder_attn.v_proj", "encoder_attn.out_proj"]

    def __init__(
        self,
        whisper_name: str       = "openai/whisper-medium.en",
        n_input: int            = 256,
        d_model: int            = 512,
        nhead: int              = 8,
        num_layers: int         = 4,
        d_ff: int               = 2048,
        conv_kernel_size: int   = 31,
        stem_kernel: int        = 32,
        stem_stride: int        = 4,
        dropout: float          = 0.1,
        spatial_attention: bool = True,
        spec_augment: bool      = False,
        n_sessions: int         = 24,
        lora_r: int             = 16,
        lora_alpha: int         = 32,
        lora_dropout: float     = 0.1,
        cross_attn_only: bool   = False,
        freeze_whisper: bool    = False,
        freeze_encoder: bool    = False,
    ):
        super().__init__()

        self.encoder = ConformerEncoder(
            n_input=n_input, d_model=d_model, nhead=nhead,
            num_layers=num_layers, d_ff=d_ff,
            conv_kernel_size=conv_kernel_size,
            stem_kernel=stem_kernel, stem_stride=stem_stride,
            dropout=dropout, spatial_attention=spatial_attention,
            spec_augment=spec_augment, n_sessions=n_sessions,
        )

        whisper_config = WhisperConfig.from_pretrained(whisper_name)
        self.whisper_d_model = whisper_config.d_model  # 1024 for medium

        # Projector: Conformer d_model (512) → Whisper d_model (1024).
        # LayerNorm aligns ECoG token scale with Whisper's decoder embedding scale.
        self.projector = nn.Sequential(
            nn.Linear(d_model, self.whisper_d_model),
            nn.LayerNorm(self.whisper_d_model),
        )

        base_whisper = WhisperForConditionalGeneration.from_pretrained(whisper_name)
        # Multilingual models need explicit language="en" in generate() to skip
        # auto language-detection, which tries to run the encoder we've bypassed.
        # medium.en vocab_size=51864; multilingual models have a larger vocab.
        self._is_multilingual = whisper_config.vocab_size != 51864

        # No task_type: PeftModel.forward() passes kwargs straight through,
        # avoiding the input_ids=None injection from PeftModelForSeq2SeqLM
        # that conflicts with decoder's input_ids kwarg via **kwargs propagation.
        lora_modules = self.LORA_CROSS_ATTN_MODULES if cross_attn_only else self.LORA_TARGET_MODULES
        self.whisper = get_peft_model(base_whisper, LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=lora_modules, bias="none",
        ))
        self.whisper.enable_input_require_grads()

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        if freeze_whisper:
            for p in self.whisper.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        ecog: torch.Tensor,
        ecog_lengths: torch.Tensor,
        input_ids: torch.Tensor,          # decoder_input_ids (teacher forcing)
        attention_mask: torch.Tensor,     # text padding mask (decoder side)
        labels: torch.Tensor,             # CE targets (-100 on forced prefix positions)
        session_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ecog_memory, ecog_len = self._encode_ecog(ecog, ecog_lengths, session_ids)
        B, T_prime, _ = ecog_memory.shape
        ecog_mask = self._ecog_mask(B, T_prime, ecog_len, ecog.device)

        out = self.whisper(
            encoder_outputs=BaseModelOutput(last_hidden_state=ecog_memory),
            attention_mask=ecog_mask,
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask,
            labels=labels,
        )
        return out.loss

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
        num_beams: int = 1,
        session_ids: torch.Tensor | None = None,
    ) -> list[str]:
        ecog_memory, ecog_len = self._encode_ecog(ecog, ecog_lengths, session_ids)
        B, T_prime, _ = ecog_memory.shape
        ecog_mask = self._ecog_mask(B, T_prime, ecog_len, ecog.device)

        lang_kwargs = {"language": "en"} if self._is_multilingual else {}
        generated = self.whisper.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=ecog_memory),
            attention_mask=ecog_mask,
            max_new_tokens=max_new_tokens,
            max_length=None,  # clear Whisper's default 448 to suppress the conflicting-limits warning
            num_beams=num_beams,
            do_sample=False,
            **lang_kwargs,
        )
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [t.strip() for t in texts]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ecog_mask(self, B, T_prime, ecog_len, device):
        if ecog_len is not None:
            return (torch.arange(T_prime, device=device).unsqueeze(0)
                    < ecog_len.unsqueeze(1)).long()
        return torch.ones(B, T_prime, dtype=torch.long, device=device)

    def _encode_ecog(self, ecog, ecog_lengths, session_ids=None):
        enc_out, enc_len = self.encoder(ecog.float(), ecog_lengths, session_ids)
        projected = self.projector(enc_out)
        # Cast to Whisper's dtype (large-v3 loads as fp16; medium.en as fp32).
        whisper_dtype = next(self.whisper.parameters()).dtype
        return projected.to(whisper_dtype), enc_len

    def build_per_session_norm(self, session_stats):
        self.encoder.build_per_session_norm(session_stats)

    def print_trainable_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")

    @classmethod
    def from_pretrained(cls, whisper_name: str, **kwargs) -> "WhisperBCIModel":
        return cls(whisper_name=whisper_name, **kwargs)
