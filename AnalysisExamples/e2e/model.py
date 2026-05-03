"""E2E BCI Model — LLaVA-style decoder-only.

Pipeline:
    ECoG (B, T, 256)
        ↓  ConformerEncoder      → (B, T', d_model)
        ↓  Linear + LayerNorm    → (B, T', llm_dim)   norm-matched to LLM embedding scale
        ↓  concat text embs      → (B, T'+text_len, llm_dim)
        ↓  Decoder-only LLM      → logits on text positions only

Generation uses Qwen3.5 no-thinking mode: ECoG prefix is followed by an empty
think block (<think>\\n</think>\\n) so the LLM immediately generates the
transcription without chain-of-thought reasoning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType

from .conformer_pt import ConformerEncoder


class E2EBCIModel(nn.Module):
    """End-to-end BCI decoder: Conformer + projector + decoder-only LLM."""

    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

    def __init__(
        self,
        llm_name_or_path: str,
        n_input: int           = 256,
        d_model: int           = 512,
        nhead: int             = 8,
        num_layers: int        = 4,
        d_ff: int              = 2048,
        conv_kernel_size: int  = 31,
        stem_kernel: int       = 32,
        stem_stride: int       = 4,
        dropout: float         = 0.1,
        spatial_attention: bool = True,
        spec_augment: bool     = False,
        n_sessions: int        = 24,
        lora_r: int            = 16,
        lora_alpha: int        = 32,
        lora_dropout: float    = 0.05,
        freeze_llm: bool       = False,
        freeze_encoder: bool   = False,
        label_smoothing: float = 0.0,
        torch_dtype            = torch.bfloat16,
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

        self.label_smoothing = label_smoothing
        llm_config = AutoConfig.from_pretrained(llm_name_or_path, trust_remote_code=True)
        self.llm_dim = getattr(llm_config, "hidden_size", None) or llm_config.text_config.hidden_size

        base_llm = AutoModelForCausalLM.from_pretrained(
            llm_name_or_path, trust_remote_code=True, torch_dtype=torch_dtype,
        )
        self.llm = get_peft_model(base_llm, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=self.LORA_TARGET_MODULES, bias="none",
        ))
        self.llm.enable_input_require_grads()

        # Qwen3.5 uses a hybrid architecture (GatedDeltaNet linear attention +
        # standard SDPA). The GatedDeltaNet fallback path (no flash-linear-attention
        # installed) produces float32 hidden states that crash against bfloat16 weight
        # matrices. A pre-hook forces bfloat16 on every GatedDeltaNet module.
        def _bf16_hook(module, args):
            return tuple(
                a.to(torch_dtype) if isinstance(a, torch.Tensor) and a.is_floating_point() else a
                for a in args
            )
        for module in self.llm.modules():
            if type(module).__name__ == "GatedDeltaNet":
                module.register_forward_pre_hook(_bf16_hook)

        # Linear + LayerNorm projector.
        # LayerNorm ensures projected ECoG tokens have the same scale as text
        # token embeddings — without it the LLM ignores ECoG (19x norm mismatch).
        self.projector = nn.Sequential(
            nn.Linear(d_model, self.llm_dim),
            nn.LayerNorm(self.llm_dim),
        ).to(torch_dtype)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        ecog: torch.Tensor,
        ecog_lengths: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        session_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ecog_embs, ecog_len = self._encode_ecog(ecog, ecog_lengths, session_ids)
        text_embs = self._text_embeddings(input_ids)
        inputs_embeds, full_attn, _ = self._concat_inputs(
            ecog_embs, ecog_len, text_embs, attention_mask, labels
        )
        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=full_attn)

        # input_ids has L tokens; labels = input_ids[1:] has L-1 tokens.
        # Slice T_prime:-1 to get exactly L-1 logits aligned with labels.
        T_prime = ecog_embs.shape[1]
        text_logits = out.logits[:, T_prime:-1, :]
        return F.cross_entropy(
            text_logits.reshape(-1, text_logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            label_smoothing=self.label_smoothing,
        )

    # ------------------------------------------------------------------
    # Generation
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
        # Wrap everything in autocast — Qwen3.5's linear attention has internal ops
        # (RMSNorm, rotary embeddings) that produce float32 intermediates without it,
        # which then crash against bfloat16 weight matrices in in_proj_qkv.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ecog_embs, ecog_len = self._encode_ecog(ecog, ecog_lengths, session_ids)
            B, T_prime, _ = ecog_embs.shape

            ecog_attn = torch.arange(T_prime, device=ecog.device).unsqueeze(0) < \
                        (ecog_len.unsqueeze(1) if ecog_len is not None
                         else torch.full((B, 1), T_prime, device=ecog.device))

            # Seed with empty think block: <think>\n</think>\n
            # Equivalent to enable_thinking=False — model skips CoT and answers directly.
            seed = self._no_think_seed(tokenizer, ecog.device)        # (S,)
            seed_emb = self._text_embeddings(seed.unsqueeze(0).expand(B, -1))

            inputs_embeds  = torch.cat([ecog_embs, seed_emb], dim=1)
            attention_mask = torch.cat(
                [ecog_attn.long(),
                 torch.ones(B, seed.shape[0], dtype=torch.long, device=ecog.device)],
                dim=1,
            )

            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            eos_id = im_end_id if im_end_id != tokenizer.unk_token_id else tokenizer.eos_token_id

            generated = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                eos_token_id=eos_id,
                pad_token_id=eos_id,
                do_sample=False,
                repetition_penalty=1.2,
            )
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [t.strip() for t in texts]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _no_think_seed(self, tokenizer, device) -> torch.Tensor:
        """Token IDs for <think>\\n</think>\\n (Qwen3.5 no-thinking seed)."""
        think_id     = tokenizer.convert_tokens_to_ids("<think>")
        end_think_id = tokenizer.convert_tokens_to_ids("</think>")
        nl           = tokenizer.encode("\n", add_special_tokens=False)
        return torch.tensor([think_id] + nl + [end_think_id] + nl,
                            dtype=torch.long, device=device)

    def _encode_ecog(self, ecog, ecog_lengths, session_ids=None):
        enc_out, enc_len = self.encoder(ecog.float(), ecog_lengths, session_ids)
        return self.projector(enc_out.to(self.projector[0].weight.dtype)), enc_len

    def build_per_session_norm(self, session_stats):
        self.encoder.build_per_session_norm(session_stats)

    def _text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        base = self.llm.get_base_model()
        if hasattr(base, "model") and hasattr(base.model, "embed_tokens"):
            return base.model.embed_tokens(input_ids)
        return base.get_input_embeddings()(input_ids)

    def _concat_inputs(self, ecog_embs, ecog_len, text_embs, text_attn, labels):
        B, T_prime, _ = ecog_embs.shape
        if ecog_len is not None:
            ecog_attn = (torch.arange(T_prime, device=ecog_embs.device).unsqueeze(0)
                         < ecog_len.unsqueeze(1)).long()
        else:
            ecog_attn = torch.ones(B, T_prime, dtype=torch.long, device=ecog_embs.device)

        inputs_embeds = torch.cat([ecog_embs, text_embs], dim=1)
        attention_mask = torch.cat([ecog_attn, text_attn], dim=1)
        ecog_ignore = torch.full((B, T_prime), -100, dtype=torch.long, device=ecog_embs.device)
        full_labels = torch.cat([ecog_ignore, labels], dim=1)
        return inputs_embeds, attention_mask, full_labels

    def print_trainable_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")

    @classmethod
    def from_pretrained(cls, llm_name_or_path: str, **kwargs) -> "E2EBCIModel":
        return cls(llm_name_or_path=llm_name_or_path, **kwargs)
