"""PyTorch Conformer encoder for E2E BCI decoding.

Input:  (B, T, 256)  — 256-channel ECoG features, 20ms bins
Output: (B, T', d_model) — contextual embeddings, T' = (T-31)//4 + 1 ≈ T/4
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

def sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                    -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)  # (1, max_len, d_model)


# ---------------------------------------------------------------------------
# Spatial attention (cross-channel gating on raw electrode space)
# ---------------------------------------------------------------------------

class SpatialAttention(nn.Module):
    """Self-attention across electrode channels, produces a per-channel gate.

    Operates on raw input (B, T, C=256) before temporal subsampling so each
    electrode's identity is still preserved.  Learnable channel embeddings let
    the attention know which electrode is which, enabling position-specific
    spatial patterns (matches the TF implementation in models.py).
    """

    def __init__(self, n_channels: int = 256, d_attn: int = 64, nhead: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.proj    = nn.Linear(1, d_attn)
        self.attn    = nn.MultiheadAttention(d_attn, nhead, dropout=dropout,
                                              batch_first=True)
        self.norm    = nn.LayerNorm(d_attn)
        self.out     = nn.Linear(d_attn, 1)
        self.ch_emb  = nn.Parameter(torch.randn(1, n_channels, d_attn) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        se = x.mean(dim=1, keepdim=True).transpose(1, 2)  # (B, C, 1)
        h  = self.proj(se) + self.ch_emb                  # (B, C, d_attn)
        attn_out, _ = self.attn(h, h, h)
        h  = self.norm(h + attn_out)                       # (B, C, d_attn)
        gate = torch.sigmoid(self.out(h)).squeeze(-1)      # (B, C)
        return x * gate.unsqueeze(1)                       # (B, T, C)


# ---------------------------------------------------------------------------
# SpecAugment
# ---------------------------------------------------------------------------

class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param: int = 27, time_mask_param: int = 10,
                 n_freq_masks: int = 2, n_time_masks: int = 2):
        super().__init__()
        self.fmp = freq_mask_param
        self.tmp = time_mask_param
        self.nf  = n_freq_masks
        self.nt  = n_time_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        if not self.training:
            return x
        B, T, F = x.shape
        # Build mask separately to avoid in-place ops on autograd tensors.
        mask = torch.ones(B, T, F, dtype=x.dtype, device=x.device)
        for _ in range(self.nf):
            f  = torch.randint(0, min(self.fmp, F), (1,)).item()
            f0 = torch.randint(0, max(F - f, 1), (1,)).item().claude
            mask[:, :, f0:f0 + f] = 0
        for _ in range(self.nt):
            t  = torch.randint(0, min(self.tmp, T), (1,)).item()
            t0 = torch.randint(0, max(T - t, 1), (1,)).item()
            mask[:, t0:t0 + t, :] = 0
        return x * mask


# ---------------------------------------------------------------------------
# Conformer sub-modules
# ---------------------------------------------------------------------------

class ConformerConvModule(nn.Module):
    """LayerNorm → Pointwise → GLU → Depthwise → BatchNorm → Swish → Pointwise → Dropout"""

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for same padding"
        self.norm  = nn.LayerNorm(d_model)
        self.pw1   = nn.Linear(d_model, 2 * d_model)
        self.dw    = nn.Conv1d(d_model, d_model, kernel_size,
                               padding=kernel_size // 2, groups=d_model, bias=False)
        self.bn    = nn.BatchNorm1d(d_model)
        self.pw2   = nn.Linear(d_model, d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        h = self.norm(x)
        h = self.pw1(h)                        # (B, T, 2*d_model)
        h1, h2 = h.chunk(2, dim=-1)
        h = h1 * torch.sigmoid(h2)            # GLU: (B, T, d_model)
        h = h.transpose(1, 2)                  # (B, d_model, T)
        h = self.dw(h)
        h = self.bn(h)
        h = F.silu(h)                          # Swish
        h = h.transpose(1, 2)                  # (B, T, d_model)
        h = self.pw2(h)
        return self.drop(h)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerBlock(nn.Module):
    """FFN(½) → MHSA → ConvModule → FFN(½) → LayerNorm"""

    def __init__(self, d_model: int, nhead: int, d_ff: int,
                 conv_kernel_size: int = 31, dropout: float = 0.1,
                 attn_dropout: float = 0.0):
        super().__init__()
        self.ffn1   = FeedForward(d_model, d_ff, dropout)
        self.norm_a = nn.LayerNorm(d_model)
        self.attn   = nn.MultiheadAttention(d_model, nhead, dropout=attn_dropout,
                                             batch_first=True)
        self.drop_a = nn.Dropout(dropout)
        self.conv   = ConformerConvModule(d_model, conv_kernel_size, dropout)
        self.ffn2   = FeedForward(d_model, d_ff, dropout)
        self.norm_f = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        h, _ = self.attn(self.norm_a(x), self.norm_a(x), self.norm_a(x),
                          key_padding_mask=key_padding_mask)
        x = x + self.drop_a(h)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.norm_f(x)


# ---------------------------------------------------------------------------
# Conv stem for temporal subsampling
# ---------------------------------------------------------------------------

class ConvStem(nn.Module):
    """1-D conv stem that stacks/subsamples temporal frames.

    Equivalent to the TF extract_patches + input_proj used in models.py but
    implemented as a learned conv so gradients flow through the subsampling.

    kernel_size=32, stride=4 gives T' = (T - 32) // 4 + 1  ≈  T / 4.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 32, stride: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, bias=False)
        self.norm = nn.LayerNorm(out_channels)
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C_in)
        x = x.transpose(1, 2)               # (B, C_in, T)
        x = self.conv(x)                    # (B, out_channels, T')
        x = x.transpose(1, 2)              # (B, T', out_channels)
        return self.norm(x)

    def output_len(self, T: int) -> int:
        return (T - self.kernel_size) // self.stride + 1


# ---------------------------------------------------------------------------
# Full Conformer encoder
# ---------------------------------------------------------------------------

class ConformerEncoder(nn.Module):
    """PyTorch Conformer encoder.

    Replaces the TF ConformerEncoder from models.py.  Outputs (B, T', d_model)
    feature embeddings instead of CTC logits — the output head lives in the
    E2E model wrapper.

    Architecture:
        [SpatialAttention] → ConvStem → scale → pos_enc → [SpecAugment]
            → N × ConformerBlock → LayerNorm
    """

    def __init__(
        self,
        n_input: int       = 256,
        d_model: int       = 512,
        nhead: int         = 8,
        num_layers: int    = 4,
        d_ff: int          = 2048,
        conv_kernel_size: int = 31,
        stem_kernel: int   = 32,
        stem_stride: int   = 4,
        dropout: float     = 0.1,
        attn_dropout: float = 0.0,
        max_seq_len: int   = 2000,
        spatial_attention: bool = True,
        spatial_attn_dim: int  = 64,
        spatial_attn_heads: int = 4,
        spec_augment: bool  = False,
        freq_mask_param: int = 27,
        time_mask_param: int = 10,
    ):
        super().__init__()
        self.d_model = d_model

        self.spatial_attn = (
            SpatialAttention(n_input, spatial_attn_dim, spatial_attn_heads, dropout)
            if spatial_attention else None
        )
        self.stem = ConvStem(n_input, d_model, stem_kernel, stem_stride)
        self.spec_aug = (
            SpecAugment(freq_mask_param, time_mask_param)
            if spec_augment else None
        )

        self.register_buffer(
            "pos_enc", sinusoidal_encoding(max_seq_len, d_model), persistent=False
        )
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, nhead, d_ff, conv_kernel_size,
                           dropout, attn_dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x:       (B, T, 256)
            lengths: (B,) original time-step counts before padding (optional)
        Returns:
            out:     (B, T', d_model)
            out_len: (B,) subsampled lengths, or None if lengths was None
        """
        if self.spatial_attn is not None:
            x = self.spatial_attn(x)

        x = self.stem(x)                              # (B, T', d_model)

        # Scale by sqrt(d_model) as in the original paper
        x = x * math.sqrt(self.d_model)

        T_prime = x.size(1)
        x = x + self.pos_enc[:, :T_prime, :].to(x.device)
        x = self.pos_drop(x)

        if self.spec_aug is not None:
            x = self.spec_aug(x)

        # Build key_padding_mask from subsampled lengths
        key_padding_mask = None
        out_len = None
        if lengths is not None:
            out_len = self.stem.output_len(lengths.clamp(min=self.stem.kernel_size))
            out_len = out_len.clamp(min=1, max=T_prime)
            key_padding_mask = torch.arange(T_prime, device=x.device).unsqueeze(0) >= \
                               out_len.unsqueeze(1)  # True = ignored

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        return self.norm(x), out_len
