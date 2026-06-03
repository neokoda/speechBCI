"""Slide variant of Gambar III.4 (arsitektur E2E gaya LLaVA).

Thesis keeps the compact `fig_e2e_llava.png`; this writes `fig_e2e_llava_h.png`
for the slide, where:
  - the gray + red top blocks (ECoG, Conformer encoder, Proyektor) are single-line
    and widened to span the full diagram width, filling the empty space to their
    right;
  - the flow then splits into Token ECoG (left) + Token teks (right), concatenates,
    and runs through the decoder-only LM (LoRA) -> LM head.

Built at 1 data unit = 1 inch so the wide single-line boxes hold their text.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))

BLUE_F, BLUE_E = "#e8f0fe", "#1a73e8"
RED_F,  RED_E  = "#fce8e6", "#d93025"
GRN_F,  GRN_E  = "#e6f4ea", "#188038"
GRY_F,  GRY_E  = "#f1f3f4", "#5f6368"

MAIN = 14
LINE_H = 0.25


def box(cx, cy, w, h, text, fc, ec):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.01,rounding_size=0.05",
                 linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=2))
    lines = text.split("\n")
    y0 = cy + (len(lines) - 1) * LINE_H / 2
    for i, ln in enumerate(lines):
        italic = ln.startswith("@i@")
        ln = ln[3:] if italic else ln
        ax.text(cx, y0 - i * LINE_H, ln, ha="center", va="center", fontsize=MAIN,
                zorder=3, fontstyle="italic" if italic else "normal")


def arrow(p1, p2, ec="#3c4043", ls="-", lw=1.5):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                 linewidth=lw, color=ec, zorder=1, shrinkA=2, shrinkB=2, linestyle=ls))


fig, ax = plt.subplots(figsize=(10, 9), dpi=100)

XL, XR = -1.5, 1.5
WCOL = 2.75
FW = 5          # full width of the top single-line blocks
H1 = 0.5         # one-line height
H2 = 0.75         # two-line height
GAP = 0.5

# --- top: full-width single-line blocks (fill the right-side space) ---
y_ecog = 0.0
y_conf = y_ecog - (H1 + GAP)
y_proj = y_conf - (H1 + GAP)
box(0, y_ecog, FW, H1, "ECoG  (B, T, 256)", GRY_F, GRY_E)
box(0, y_conf, FW, H2, "Conformer encoder\n(d=512, spatial attention, subsample 4x)", RED_F, RED_E)
box(0, y_proj, FW, H1, "Proyektor  (Linear 512 -> d_LM, + LayerNorm)", RED_F, RED_E)
arrow((0, y_ecog - H1 / 2), (0, y_conf + H2 / 2))
arrow((0, y_conf - H1 / 2), (0, y_proj + H1 / 2))

# --- split into Token ECoG (left) + Token teks (right) ---
y_tok = y_proj - (H1 / 2 + GAP + H2 / 2)
box(XL, y_tok, WCOL, H1, "Token ECoG (B, T', d_LM)", GRN_F, GRN_E)
box(XR, y_tok, WCOL, H1, "Token teks", BLUE_F, BLUE_E)
arrow((XL, y_proj - H1 / 2), (XL, y_tok + H1 / 2))   # Proyektor -> Token ECoG

# --- concat ---
y_cat = y_tok - (H1 / 2 + GAP + H1 / 2)
box(0, y_cat, 5.6, H1, "Konkatenasi [ Token ECoG | Token teks ]", BLUE_F, BLUE_E)
arrow((XL, y_tok - H1 / 2), (-1.3, y_cat + H1 / 2))
arrow((XR, y_tok - H1 / 2), (1.3, y_cat + H1 / 2))

# --- decoder-only LM -> LM head ---
y_lm = y_cat - (H2 / 2 + GAP + H1 / 2)
box(0, y_lm, 5.6, H1, "Decoder-only LM (LoRA)", BLUE_F, BLUE_E)
arrow((0, y_cat - H1 / 2), (0, y_lm + H1 / 2))

y_head = y_lm - (H1 / 2 + GAP + H1 / 2)
box(0, y_head, 5.6, H1, "LM head -> teks akhir", GRN_F, GRN_E)
arrow((0, y_lm - H1 / 2), (0, y_head + H1 / 2))

xmin, xmax = -FW / 2 - 0.4, FW / 2 + 0.4
ymin, ymax = y_head - H1 / 2 - 0.4, y_ecog + H1 / 2 + 0.4
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.axis("off")
ax.set_position([0, 0, 1, 1])
fig.set_size_inches(xmax - xmin, ymax - ymin)
out = os.path.join(HERE, "fig_e2e_llava_h.png")
fig.savefig(out, dpi=200, facecolor="white")
plt.close(fig)
print("wrote", out)
