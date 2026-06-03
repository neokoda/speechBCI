"""Landscape (two-row) variant of Gambar III.3 (arsitektur dua tahap) for slides.

Thesis keeps the tall vertical `fig_dua_tahap.png`; this writes a separate
`fig_dua_tahap_h.png`:
  top row    : gray + yellow (Fitur -> dekoder fonem -> Probabilitas fonem), L->R
               with a bracket marking the yellow blocks as "Tahap 1 (model akustik)"
  bottom row : blue + green  (WFST -> n-best -> rescoring -> Teks akhir), L->R
               with a bracket marking the blue blocks as "Tahap 2 (model bahasa)"
A diagonal carries from the end of the top row to the start of the bottom row.

Each block has a DYNAMIC width (measured from its label) and a FIXED one-line
height. Box widths/heights are in inches (1 data unit = 1 inch), so text never
overflows regardless of label length.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))

BLUE_F, BLUE_E = "#e8f0fe", "#1a73e8"
YEL_F,  YEL_E  = "#fef7e0", "#8b5f00"
GRN_F,  GRN_E  = "#e6f4ea", "#188038"
GRY_F,  GRY_E  = "#f1f3f4", "#5f6368"

MAIN = 14
H = 0.62          # fixed one-line box height (inches)
PAD = 0.24        # horizontal text padding per side (inches)
GAP = 0.75        # gap between boxes in a row (inches)
TOP_Y, BOT_Y = 0.0, -1

fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
fig.canvas.draw()
REND = fig.canvas.get_renderer()


def text_w(s):
    """Rendered width of label in inches (font size fixed in points)."""
    t = ax.text(0, 0, s, fontsize=MAIN)
    w = t.get_window_extent(REND).width / fig.dpi
    t.remove()
    return w


def box(cx, cy, w, txt, fc, ec):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - H / 2), w, H,
                 boxstyle="round,pad=0.01,rounding_size=0.05",
                 linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(cx, cy, txt, ha="center", va="center", fontsize=MAIN, zorder=3)


def arrow(p1, p2, ec="#3c4043"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                 linewidth=1.5, color=ec, zorder=1, shrinkA=2, shrinkB=2))


def layout(boxes, y, x0=0.0):
    placed, x = [], x0
    for txt, fc, ec in boxes:
        w = text_w(txt) + 2 * PAD
        placed.append((x + w / 2, y, w, txt, fc, ec))
        x += w + GAP
    return placed


def hbracket(x0, x1, y, label, color, above=True):
    tick = -0.14 if above else 0.14
    ax.plot([x0, x1], [y, y], color=color, lw=1.6, zorder=1)
    ax.plot([x0, x0], [y, y + tick], color=color, lw=1.6, zorder=1)
    ax.plot([x1, x1], [y, y + tick], color=color, lw=1.6, zorder=1)
    ax.text((x0 + x1) / 2, y + (0.16 if above else -0.16), label,
            ha="center", va="bottom" if above else "top",
            fontsize=14, color=color)


top_boxes = [
    ("Fitur neural  T x 256", GRY_F, GRY_E),
    ("Model berbasis Transformer (dekoder fonem)", YEL_F, YEL_E),
    ("Probabilitas fonem (CTC), 40 kelas + token blank", YEL_F, YEL_E),
]
bot_boxes = [
    ("Dekode WFST 5-gram (beam search + shallow fusion)", BLUE_F, BLUE_E),
    ("Daftar n-best hipotesis", BLUE_F, BLUE_E),
    ("Rescoring LLaMA-2 7B", BLUE_F, BLUE_E),
    ("Teks akhir", GRN_F, GRN_E),
]

top = layout(top_boxes, TOP_Y)
bot = layout(bot_boxes, BOT_Y)

for p in top + bot:
    box(*p)
for i in range(len(top) - 1):
    arrow((top[i][0] + top[i][2] / 2, TOP_Y), (top[i + 1][0] - top[i + 1][2] / 2, TOP_Y))
for i in range(len(bot) - 1):
    arrow((bot[i][0] + bot[i][2] / 2, BOT_Y), (bot[i + 1][0] - bot[i + 1][2] / 2, BOT_Y))
# carry: end of top row -> start of bottom row
arrow((top[-1][0], TOP_Y - H / 2), (bot[0][0], 0.95 * BOT_Y + H / 2))

# Tahap 1 bracket over the two yellow boxes; Tahap 2 under the three blue boxes
hbracket(top[1][0] - top[1][2] / 2, top[2][0] + top[2][2] / 2,
         TOP_Y + H / 2 + 0.30, "Tahap 1 (model akustik)", YEL_E, above=True)
hbracket(bot[0][0] - bot[0][2] / 2, bot[2][0] + bot[2][2] / 2,
         BOT_Y - H / 2 - 0.30, "Tahap 2 (model bahasa)", BLUE_E, above=False)

xs = [p[0] - p[2] / 2 for p in top + bot] + [p[0] + p[2] / 2 for p in top + bot]
xmin, xmax = min(xs) - 0.4, max(xs) + 0.4
ymin, ymax = BOT_Y - H / 2 - 0.8, TOP_Y + H / 2 + 0.8
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.axis("off")
# Fill the figure and size it to the data span so 1 data unit == 1 inch,
# which keeps measured label widths consistent with the rendered boxes.
ax.set_position([0, 0, 1, 1])
fig.set_size_inches(xmax - xmin, ymax - ymin)
out = os.path.join(HERE, "fig_dua_tahap_h.png")
fig.savefig(out, dpi=200, facecolor="white")
plt.close(fig)
print("wrote", out)
