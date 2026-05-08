"""GraniteBCIModel — IBM Granite Speech 4.1-2B adapted for ECoG BCI decoding.

Pipeline:
    ECoG (B, T, 256)
        ↓  ConformerEncoder      → (B, T', 512)
        ↓  Linear + LayerNorm    → (B, T', 2048)   [Granite hidden_size]
        ↓  concat BOS emb        → (B, T'+1, 2048)
        ↓  GraniteForCausalLM    → logits on text positions only

Same LLaVA-style prefix injection as v4/v5 but with Granite 4.0 1B (40 layers,
hidden=2048, GQA) as the LM decoder instead of Qwen3.5-0.8B.

The LM weights come from ibm-granite/granite-speech-4.1-2b (.language_model),
which is a pure-attention Granite — distinct from the standalone
granite-4.0-1b-base (Mamba hybrid).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSpeechSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType

from .conformer_pt import ConformerEncoder


class GraniteBCIModel(nn.Module):
    """ECoG BCI decoder: Conformer + projector + Granite 4.0 1B (LoRA)."""

    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

    def __init__(
        self,
        granite_name: str       = "ibm-granite/granite-speech-4.1-2b",
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
        freeze_llm: bool        = False,
        freeze_encoder: bool    = False,
        torch_dtype             = torch.bfloat16,
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

        # Load only the LM from the full speech model; discard encoder + Q-Former.
        print(f"Loading Granite LM from {granite_name} ...")
        speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            granite_name, torch_dtype=torch_dtype,
        )
        base_lm = speech_model.language_model
        del speech_model

        self.granite_dim = base_lm.config.hidden_size  # 2048

        self.llm = get_peft_model(base_lm, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=self.LORA_TARGET_MODULES, bias="none",
        ))
        self.llm.enable_input_require_grads()

        self.projector = nn.Sequential(
            nn.Linear(d_model, self.granite_dim),
            nn.LayerNorm(self.granite_dim),
        ).to(torch_dtype)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Forward (training — teacher forcing)
    # ------------------------------------------------------------------

    def forward(
        self,
        ecog: torch.Tensor,
        ecog_lengths: torch.Tensor,
        input_ids: torch.Tensor,       # [BOS, tok_0, ..., tok_N]  (from dataset)
        attention_mask: torch.Tensor,  # text padding mask
        labels: torch.Tensor,          # [tok_0, ..., tok_N, EOS]  (causal shift)
        session_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ecog_embs, ecog_len = self._encode_ecog(ecog, ecog_lengths, session_ids)
        text_embs = self._text_embeddings(input_ids)
        inputs_embeds, full_attn = self._concat_inputs(
            ecog_embs, ecog_len, text_embs, attention_mask
        )
        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=full_attn)

        # logits[:, T_prime:-1] aligns with labels (text portion, causal shift)
        T_prime = ecog_embs.shape[1]
        text_logits = out.logits[:, T_prime:-1, :]
        return F.cross_entropy(
            text_logits.reshape(-1, text_logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

    # ------------------------------------------------------------------
    # Generation (inference — autoregressive)
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
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ecog_embs, ecog_len = self._encode_ecog(ecog, ecog_lengths, session_ids)
            B, T_prime, _ = ecog_embs.shape

            ecog_attn = (torch.arange(T_prime, device=ecog.device).unsqueeze(0)
                         < (ecog_len.unsqueeze(1) if ecog_len is not None
                            else torch.full((B,), T_prime, device=ecog.device).unsqueeze(1))
                         ).long()

            # Seed with BOS — matches training prefix (dataset BOS fallback)
            bos = torch.tensor(
                [tokenizer.bos_token_id or tokenizer.eos_token_id],
                dtype=torch.long, device=ecog.device,
            )
            bos_emb = self._text_embeddings(bos.unsqueeze(0).expand(B, -1))

            inputs_embeds = torch.cat([ecog_embs, bos_emb], dim=1)
            attention_mask = torch.cat(
                [ecog_attn,
                 torch.ones(B, 1, dtype=torch.long, device=ecog.device)],
                dim=1,
            )

            generated = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                repetition_penalty=1.2,
            )
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [t.strip() for t in texts]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_ecog(self, ecog, ecog_lengths, session_ids=None):
        enc_out, enc_len = self.encoder(ecog.float(), ecog_lengths, session_ids)
        return self.projector(enc_out.to(self.projector[0].weight.dtype)), enc_len

    def _text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        base = self.llm.get_base_model()
        if hasattr(base, "model") and hasattr(base.model, "embed_tokens"):
            return base.model.embed_tokens(input_ids)
        return base.get_input_embeddings()(input_ids)

    def _concat_inputs(self, ecog_embs, ecog_len, text_embs, text_attn):
        B, T_prime, _ = ecog_embs.shape
        if ecog_len is not None:
            ecog_attn = (torch.arange(T_prime, device=ecog_embs.device).unsqueeze(0)
                         < ecog_len.unsqueeze(1)).long()
        else:
            ecog_attn = torch.ones(B, T_prime, dtype=torch.long, device=ecog_embs.device)
        inputs_embeds = torch.cat([ecog_embs, text_embs], dim=1)
        attention_mask = torch.cat([ecog_attn, text_attn], dim=1)
        return inputs_embeds, attention_mask

    def build_per_session_norm(self, session_stats):
        self.encoder.build_per_session_norm(session_stats)

    def print_trainable_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")
