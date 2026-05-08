"""CanaryBCIModel — Qwen3-1.7B with pretrained Canary LoRA adapters.

Pipeline:
    ECoG (B, T, 256)
        ↓  ConformerEncoder      → (B, T', 512)
        ↓  pre_projector        → (B, T', 1024)   [random init, trainable]
        ↓  canary_projector    → (B, T', 2048)   [pretrained from Canary: Linear(1024, 2048)]
        ↓  concat [ECoG_emb | text_emb | EOS] → (B, T'+L, 2048)
        ↓  Qwen3ForCausalLM + LoRA → logits on text positions only

Key advantage: LoRA adapters (r=128, q_proj+v_proj) are loaded from
nvidia/canary-qwen-2.5b — pretrained for speech→text alignment.
Qwen3-1.7B has GrokCol support (no special hooks needed unlike Qwen3.5).

The canary_projector is pretrained (Linear 1024→2048). The pre_projector is
random-initialized and trained from scratch. Both are trainable.

Qwen3-1.7B generation uses the no-thinking seed: <think>\n\n\n\n</think>\n\n
to skip chain-of-thought and generate directly.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

from .conformer_pt import ConformerEncoder


# Path to the downloaded canary-qwen-2.5b safetensors
CANARY_CKPT = os.environ.get(
    "CANARY_CKPT",
    "/workspace/.hf_cache/hub/models--nvidia--canary-qwen-2.5b/model.safetensors"
)


class CanaryBCIModel(nn.Module):
    """ECoG BCI decoder: Conformer + pre_projector + pretrained Canary projector + Qwen3-1.7B (LoRA r=128)."""

    # Canary LoRA: q_proj + v_proj only (r=128, alpha=256)
    LORA_TARGET_MODULES = ["q_proj", "v_proj"]

    def __init__(
        self,
        qwen_name: str          = "Qwen/Qwen3-1.7B",
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
        lora_r: int             = 128,
        lora_alpha: int         = 256,
        lora_dropout: float     = 0.01,
        torch_dtype             = torch.bfloat16,
        load_canary_llm: bool   = True,
        pretrained_encoder_path: str | None = None,
        freeze_llm: bool = False,
        freeze_encoder: bool = False,
        freeze_canary_projector: bool = False,
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

        if pretrained_encoder_path:
            self._load_pretrained_encoder(pretrained_encoder_path)

        # Stage 1: 512→1024 (random init, trainable)
        self.pre_projector = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.LayerNorm(1024),
        ).to(torch_dtype)

        # Stage 2: 1024→2048 (pretrained from Canary safetensors)
        print(f"Loading pretrained canary_projector from {CANARY_CKPT} ...")
        self.canary_projector = nn.Linear(1024, 2048).to(torch_dtype)
        self._load_canary_projector()

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        if freeze_canary_projector:
            for p in self.canary_projector.parameters():
                p.requires_grad_(False)

        # Load Qwen3-1.7B base
        print(f"Loading Qwen3-1.7B base from {qwen_name} ...")
        base_lm = AutoModelForCausalLM.from_pretrained(
            qwen_name, torch_dtype=torch_dtype, trust_remote_code=True,
        )
        self.qwen_dim = base_lm.config.hidden_size  # 2048

        # Apply LoRA matching Canary's config: r=128, q_proj+v_proj
        self.llm = get_peft_model(base_lm, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=self.LORA_TARGET_MODULES, bias="none",
        ))
        self.llm.enable_input_require_grads()

        # Load pretrained LoRA weights from Canary
        if load_canary_llm:
            print("Loading pretrained LoRA weights from Canary ...")
            self._load_canary_lora()

        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_pretrained_encoder(self, path: str):
        """Load pretrained CTC encoder weights from a checkpoint.

        Supports checkpoints saved by the CTC trainer that store the encoder
        under the 'model' key with 'encoder.' prefixed names.

        Args:
            path: Path to a .pt checkpoint (e.g. experiments/ctc_4l/best/checkpoint.pt)
                  or the directory containing it (will try best/checkpoint.pt then checkpoint.pt)
        """
        import os as _os
        if _os.path.isdir(path):
            for fname in ["best/checkpoint.pt", "checkpoint.pt"]:
                candidate = _os.path.join(path, fname)
                if _os.path.exists(candidate):
                    path = candidate
                    break
        print(f"Loading pretrained CTC encoder from {path} ...")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = ckpt.get("model", ckpt.get("model_full", ckpt))

        # Strip 'encoder.' prefix — CTC checkpoints use 'encoder.xxx', our
        # ConformerEncoder.state_dict() uses 'xxx'
        encoder_sd = {}
        for k, v in model.items():
            if k.startswith("encoder."):
                encoder_sd[k[len("encoder."):]] = v

        # Verify shapes match our encoder
        our_sd = self.encoder.state_dict()
        mismatches = []
        for k, v in encoder_sd.items():
            if k not in our_sd:
                print(f"  [encoder] key not found in model: {k}")
            elif v.shape != our_sd[k].shape:
                mismatches.append(f"  [encoder] shape mismatch: {k}: ckpt={list(v.shape)} our={list(our_sd[k].shape)}")

        if mismatches:
            for m in mismatches:
                print(m)
            raise ValueError(f"Encoder checkpoint key/shape mismatch at {path}")

        self.encoder.load_state_dict(encoder_sd, strict=True)
        print(f"  Loaded {len(encoder_sd)} pretrained encoder keys (includes per-session norm stats).")

    def _load_canary_projector(self):
        """Load the pretrained 1024→2048 projector from Canary safetensors."""
        canary_sd = self._read_canary_safetensors()
        weight = canary_sd.get("perception.proj.weight")
        bias   = canary_sd.get("perception.proj.bias")
        if weight is not None:
            self.canary_projector.weight.data = weight.to(self.canary_projector.weight.dtype)
        if bias is not None:
            self.canary_projector.bias.data = bias.to(self.canary_projector.bias.dtype)
        print(f"  Loaded canary_projector: {list(self.canary_projector.weight.shape)}")

    def _load_canary_lora(self):
        """Load pretrained LoRA weights from Canary's safetensors into our PEFT model.

        Canary key format:
            llm.base_model.model.model.layers.N.self_attn.q_proj.lora_A.default.weight
            llm.base_model.model.model.layers.N.self_attn.v_proj.lora_B.default.weight

        Our PEFT model key format (Qwen3ForCausalLM + get_peft_model):
            base_model.model.layers.N.self_attn.q_proj.lora_A.weight
            base_model.model.layers.N.self_attn.v_proj.lora_B.weight

        The canary safetensor also contains frozen base weights (q_proj.base_layer)
        and frozen non-LoRA weights (k_proj, o_proj, mlp.*), but those are already
        in the Qwen3-1.7B base model from HuggingFace. We only load the LoRA parts.
        """
        canary_sd = self._read_canary_safetensors()
        peft_state = self.llm.state_dict()
        lora_key_map = {}

        for canary_key, canary_tensor in canary_sd.items():
            if "lora" not in canary_key.lower():
                continue

            # Canary:  llm.base_model.model.model.layers.N.self_attn.q_proj.lora_A.default.weight
            # PEFT:    base_model.model.model.layers.N.self_attn.q_proj.lora_A.default.weight
            # Both use .model.model.model.layers and .lora_X.default — just strip "llm."
            peft_key = canary_key.replace("llm.", "", 1)

            if peft_key in peft_state:
                lora_key_map[peft_key] = canary_tensor.to(peft_state[peft_key].dtype)
            else:
                print(f"  [LoRA] key not found in PEFT model: {peft_key}")

        # Load matched keys
        loaded = 0
        for key, tensor in lora_key_map.items():
            if tensor.shape == peft_state[key].shape:
                peft_state[key].copy_(tensor)
                loaded += 1
            else:
                print(f"  [LoRA] shape mismatch: {key}: "
                      f"canary={list(tensor.shape)} our={list(peft_state[key].shape)}")

        print(f"  Loaded {loaded}/{len(lora_key_map)} LoRA keys")

    @staticmethod
    def _read_canary_safetensors():
        """Read all tensors from Canary safetensors (cached)."""
        if not hasattr(CanaryBCIModel, "_canary_cache"):
            print(f"Reading Canary safetensors from {CANARY_CKPT} ...")
            tensors = {}
            with safe_open(CANARY_CKPT, framework="pt") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
            CanaryBCIModel._canary_cache = tensors
            print(f"  Cached {len(tensors)} tensors")
        return CanaryBCIModel._canary_cache

    # ------------------------------------------------------------------
    # Forward (training — teacher forcing)
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
        inputs_embeds, full_attn = self._concat_inputs(
            ecog_embs, ecog_len, text_embs, attention_mask
        )
        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=full_attn)

        # input_ids has L tokens; labels = input_ids[1:] has L tokens.
        # Slice T_prime:-1 to get exactly L logits aligned with labels.
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

            # Qwen3 no-thinking seed: <think>\n\n\n\n</think>\n\n
            # Equivalent to enable_thinking=False — model generates directly.
            seed = self._no_think_seed(tokenizer, ecog.device)        # (S,)
            seed_emb = self._text_embeddings(seed.unsqueeze(0).expand(B, -1))

            inputs_embeds  = torch.cat([ecog_embs, seed_emb], dim=1)
            attention_mask = torch.cat(
                [ecog_attn,
                 torch.ones(B, seed.shape[0], dtype=torch.long, device=ecog.device)],
                dim=1,
            )

            # Qwen3 uses <|im_end|> as EOS
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

    def _no_think_seed(self, tokenizer, device):
        # Seed with <|im_start|>assistant\n so model generates the transcript directly
        ims = tokenizer.convert_tokens_to_ids("<|im_start|>")
        nl  = tokenizer.encode("\n", add_special_tokens=False)
        # 151644 = <|im_start|>, assistant role, then newline
        assistant = tokenizer.convert_tokens_to_ids("assistant")
        return torch.tensor([ims, assistant, nl[0]], dtype=torch.long, device=device)

    def _encode_ecog(self, ecog, ecog_lengths, session_ids=None):
        enc_out, enc_len = self.encoder(ecog.float(), ecog_lengths, session_ids)
        dtype = self.canary_projector.weight.dtype
        enc_out = enc_out.to(dtype)
        # Two-stage: 512→1024 (trainable) then 1024→2048 (pretrained)
        pre = self.pre_projector(enc_out)
        return self.canary_projector(pre), enc_len

    def _text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        base = self.llm.get_base_model()
        if hasattr(base, "model") and hasattr(base.model, "embed_tokens"):
            return base.model.embed_tokens(input_ids)
        if hasattr(base, "embed_tokens"):
            return base.embed_tokens(input_ids)
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

    @classmethod
    def from_pretrained(cls, qwen_name: str, **kwargs) -> "CanaryBCIModel":
        return cls(qwen_name=qwen_name, **kwargs)
