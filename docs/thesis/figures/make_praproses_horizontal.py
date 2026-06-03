"""Landscape (two-row, Z-flow) variant of Gambar III.2 for the slide deck.

The thesis keeps the tall vertical `fig_praproses.png`; this writes a separate
`fig_praproses_h.png` laid out as a Z:
  top row   : gray + blue   (Sinyal -> Fitur -> Matriks -> Normalisasi), L->R
  diagonal  : carry from top-right down to bottom-left
  bottom row: green + yellow (Gaussian/augmentasi -> Input ke model), L->R

Style (colors/fonts) copied from make_figures.py to match the other figures.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))

BLUE_F, BLUE_E = "#e8f0fe", "#1a73e8"
YEL_F,  YEL_E  = "#fef7e0", "#f9ab00"
GRN_F,  GRN_E  = "#e6f4ea", "#188038"
GRY_F,  GRY_E  = "#f1f3f4", "#5f6368"

MAIN = 13
LINE_H = 0.42


def box(ax, x, y, w, h, text, fc, ec, fs=MAIN):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.01,rounding_size=0.06",
                 linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=2))
    lines = text.split("\n")
    y0 = y + (len(lines) - 1) * LINE_H / 2
    for i, ln in enumerate(lines):
        italic = ln.startswith("@i@")
        ln = ln[3:] if italic else ln
        ax.text(x, y0 - i * LINE_H, ln, ha="center", va="center", fontsize=fs,
                zorder=3, fontstyle="italic" if italic else "normal")


def bh(text, base=1.0):
    return base + 0.55 * text.count("\n")


def arrow(ax, p1, p2, ec="#3c4043", lw=1.5):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                 linewidth=lw, color=ec, zorder=1, shrinkA=2, shrinkB=2))


W = 5
DX = 5.8          # horizontal spacing between box centres
TOP_Y, BOT_Y = 0.0, -3

# top row, left -> right (gray + 3 blue)
top = [
    (0 * DX, "Sinyal ECoG mentah\n128 kanal elektroda", GRY_F, GRY_E),
    (1 * DX, "Fitur per bin 20 ms\n@i@128 threshold crossings\n@i@+ 128 spike band power", BLUE_F, BLUE_E),
    (2 * DX, "Matriks fitur  T x 256", BLUE_F, BLUE_E),
    (3 * DX, "Normalisasi z-score\nper sesi", BLUE_F, BLUE_E),
]
# bottom row, left -> right (green + yellow), aligned under the first two top boxes
bottom = [
    (0.5 * DX, "Gaussian smoothing +\naugmentasi (saat latih)\n@i@white noise, constant offset", GRN_F, GRN_E),
    (2 * DX, "Input ke model\n(dua tahap & end-to-end)", YEL_F, YEL_E),
]

fig, ax = plt.subplots(figsize=(11.0, 4.6))

for x, txt, fc, ec in top:
    box(ax, x, TOP_Y, W, bh(txt), txt, fc, ec)
for x, txt, fc, ec in bottom:
    box(ax, x, BOT_Y, W * 1.5, bh(txt), txt, fc, ec)

# top arrows L->R
for i in range(len(top) - 1):
    arrow(ax, (top[i][0] + W / 2, TOP_Y), (top[i + 1][0] - W / 2, TOP_Y))
# Z diagonal: top-right box down to bottom-left box
arrow(ax, (top[-1][0], TOP_Y - bh(top[-1][1]) / 2),
          (bottom[0][0], 0.975 * BOT_Y + bh(bottom[0][1]) / 2))
# bottom arrows L->R
for i in range(len(bottom) - 1):
    arrow(ax, (bottom[i][0] + 1.5 * W / 2, BOT_Y), (bottom[i + 1][0] - 1.5 * W / 2, BOT_Y))

ax.set_xlim(-W / 2 - 0.5, 3 * DX + W / 2 + 0.5)
ax.set_ylim(BOT_Y - bh(bottom[0][1]) / 2 - 0.5, TOP_Y + bh(top[1][1]) / 2 + 0.5)
ax.axis("off")
ax.set_aspect("equal")
fig.tight_layout(pad=0.2)
out = os.path.join(HERE, "fig_praproses_h.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote", out)
