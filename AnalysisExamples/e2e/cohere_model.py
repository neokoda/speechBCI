"""CohereBCI Model — Cohere Transcribe cross-attention E2E BCI decoder.

Pipeline (mirrors AnalysisExamples/e2e/whisper_model.py):
    ECoG (B, T, 256)
        ↓  ConformerEncoder           → (B, T', d_model=512)
        ↓  Linear + LayerNorm         → (B, T', 1024)            [Cohere decoder hidden]
        = ECoG memory  ──────────────────────────────────────────────┐
                                                                     │ K, V (cross-attn)
    Text (teacher-forced at train / autoregressive at inference):    │
        ↓  Cohere token embedding   → (B, L, 1024)                  │
        ↓  Causal self-attn         (text → text only, causal)      │
        ↓  Cross-attention          (Q=text, K=V=ECoG memory) ←─────┘
        ↓  FFN
        × 8 Cohere transformer decoder layers (pre-trained, LoRA'd)
        ↓  log_softmax head        → logits (B, L, 16384)

We bypass the Cohere audio Conformer encoder entirely (it expects 128-bin Mel
features). The decoder is fed our ECoG memory via `encoder_outputs` so the
model's forward() skips the encoder (configuration_cohere_asr.py path).

LoRA target modules: Cohere's DecoderAttention uses query_net / key_net /
value_net / out_projection (not the standard q_proj / k_proj naming).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForSpeechSeq2Seq
from transformers.modeling_outputs import BaseModelOutput
from peft import get_peft_model, LoraConfig

from .conformer_pt import ConformerEncoder


COHERE_REPO_DEFAULT = "CohereLabs/cohere-transcribe-03-2026"


class CohereBCIModel(nn.Module):
    """End-to-end BCI decoder: Conformer + projector + Cohere decoder cross-attention."""

    # Cohere decoder attention modules (different names from Whisper/Qwen)
    LORA_TARGET_MODULES     = ["query_net", "key_net", "value_net", "out_projection"]
    # Cross-attention only: same submodule names but only in the second_sub_layer of
    # each TransformerDecoderLayer. PEFT's target_modules matches by leaf name, so
    # we can't isolate the cross-attn block by name alone — use LORA_TARGET_MODULES
    # and accept LoRA on both self- and cross-attn projections (matches Whisper default).

    def __init__(
        self,
        cohere_repo: str        = COHERE_REPO_DEFAULT,
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
        freeze_cohere: bool     = False,
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

        base = AutoModelForSpeechSeq2Seq.from_pretrained(
            cohere_repo, trust_remote_code=True,
        )
        self.cohere_decoder_hidden = base.decoder_hidden_size  # 1024

        # Projector: Conformer d_model (512) → Cohere decoder hidden (1024).
        self.projector = nn.Sequential(
            nn.Linear(d_model, self.cohere_decoder_hidden),
            nn.LayerNorm(self.cohere_decoder_hidden),
        )

        # Bypass Cohere's audio encoder entirely — we feed ECoG memory via
        # encoder_outputs, so the encoder branch in forward() is never taken.
        # Replace with a placeholder; HF generate() inspects
        # `model.encoder.main_input_name`, so the stub must expose that attr.
        del base.encoder
        class _EncoderStub(nn.Module):
            main_input_name = "input_features"
            def forward(self, *args, **kwargs):
                raise RuntimeError(
                    "CohereBCIModel.encoder is a stub; pass encoder_outputs to "
                    "forward()/generate() so the encoder branch is skipped."
                )
        base.encoder = _EncoderStub()
        # ECoG memory is already at the decoder hidden size (1024), so skip the
        # encoder→decoder projection that maps 1280→1024.
        base.encoder_decoder_proj = None

        self.cohere = get_peft_model(base, LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=self.LORA_TARGET_MODULES, bias="none",
        ))
        # Ensure encoder_outputs flow through gradients to our projector/encoder.
        self.cohere.enable_input_require_grads()

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        if freeze_cohere:
            for p in self.cohere.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Forward (training): cross-entropy on text tokens.
    # ------------------------------------------------------------------

    def forward(
        self,
        ecog: torch.Tensor,
        ecog_lengths: torch.Tensor,
        input_ids: torch.Tensor,          # decoder_input_ids (teacher forcing)
        attention_mask: torch.Tensor,     # text padding mask (decoder side)
        labels: torch.Tensor,             # CE targets (-100 on forced prefix / pad)
        session_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ecog_memory, ecog_len = self._encode_ecog(ecog, ecog_lengths, session_ids)
        cross_mask = self._cross_attention_mask(ecog_memory, ecog_len)

        out = self.cohere(
            encoder_outputs=BaseModelOutput(last_hidden_state=ecog_memory),
            cross_attention_mask=cross_mask,
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
        cross_mask = self._cross_attention_mask(ecog_memory, ecog_len)

        B = ecog_memory.shape[0]
        # Minimal decoder prompt: <|startofcontext|><|startoftranscript|><|en|><|en|>
        # — matches Cohere build_prompt's first 4 tokens (the rest are options
        # for diarize/timestamp/etc that don't apply to single-language ECoG).
        decoder_start = tokenizer.convert_tokens_to_ids("<|startofcontext|>")
        bos           = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        lang          = tokenizer.convert_tokens_to_ids("<|en|>")
        prompt = torch.tensor(
            [decoder_start, bos, lang, lang],
            dtype=torch.long, device=ecog.device,
        ).unsqueeze(0).expand(B, -1)

        eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        pad_id = tokenizer.convert_tokens_to_ids("<pad>")

        generated = self.cohere.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=ecog_memory),
            cross_attention_mask=cross_mask,
            decoder_input_ids=prompt,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [t.strip() for t in texts]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_ecog(self, ecog, ecog_lengths, session_ids=None):
        enc_out, enc_len = self.encoder(ecog.float(), ecog_lengths, session_ids)
        projected = self.projector(enc_out)
        # Cast to the Cohere decoder dtype (matches the loaded model precision).
        cohere_dtype = next(self.cohere.parameters()).dtype
        return projected.to(cohere_dtype), enc_len

    def _cross_attention_mask(self, ecog_memory: torch.Tensor, ecog_len: torch.Tensor | None):
        """Build the additive cross-attention mask Cohere's decoder expects.

        Shape: (B, 1, 1, T'). Valid positions add 0; invalid add -1e9.
        Matches modeling_cohere_asr.py:850.
        """
        B, T_prime, _ = ecog_memory.shape
        device, dtype = ecog_memory.device, ecog_memory.dtype
        if ecog_len is None:
            return torch.zeros((B, 1, 1, T_prime), device=device, dtype=dtype)
        positions = torch.arange(T_prime, device=device)[None, :]
        valid = positions < ecog_len.to(device)[:, None]
        return (1.0 - valid[:, None, None, :].to(dtype=dtype)) * -1e9

    def build_per_session_norm(self, session_stats):
        self.encoder.build_per_session_norm(session_stats)

    def print_trainable_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")

    @classmethod
    def from_pretrained(cls, cohere_repo: str = COHERE_REPO_DEFAULT, **kwargs) -> "CohereBCIModel":
        return cls(cohere_repo=cohere_repo, **kwargs)
