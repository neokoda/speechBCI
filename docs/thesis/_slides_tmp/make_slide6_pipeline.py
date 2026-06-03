# -*- coding: utf-8 -*-
"""Top-down (vertical) remake of slide 6's two pipeline graphics, side by side.
Matches the deck's draw.io palette so it drops onto the white slide cleanly.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch, PathPatch
from matplotlib.path import Path

# --- palette (sampled from the existing slides) ---
BLUE_F, BLUE_E = "#DAE8FC", "#6C8EBF"   # neural feature / phoneme prob / text
YEL_F,  YEL_E  = "#FFF2CC", "#D6B656"   # learned modules (BiRNN, LM)
RED_F,  RED_E  = "#F8CECC", "#B85450"   # end-to-end model
TXT            = "#323232"
EP_RED         = "#B85450"

fig, ax = plt.subplots(figsize=(8.2, 8.0), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

BW, BH = 30, 11          # box width / height
LCX, RCX = 27, 80        # column centres

def rbox(cx, cy, label, ff, ee, italic_word=None):
    x, y = cx - BW/2, cy - BH/2
    ax.add_patch(FancyBboxPatch((x, y), BW, BH,
        boxstyle="round,pad=0,rounding_size=1.6",
        linewidth=1.4, edgecolor=ee, facecolor=ff, mutation_aspect=1))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=20,
            color=TXT, wrap=True)

def pbox(cx, cy, label, ff, ee):          # parallelogram (I/O)
    sk = 3.2
    x, y = cx - BW/2, cy - BH/2
    pts = [(x+sk, y), (x+BW, y), (x+BW-sk, y+BH), (x, y+BH)]
    ax.add_patch(Polygon(pts, closed=True, linewidth=1.4,
                          edgecolor=ee, facecolor=ff))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=20, color=TXT)

def arrow(cx, y_top, y_bot):
    ax.add_patch(FancyArrowPatch((cx, y_top), (cx, y_bot),
        arrowstyle="-|>", mutation_scale=18, linewidth=1.8, color="#666666"))

# ----- column titles -----
ax.text(LCX, 99, "Arsitektur Dua Tahap", ha="center", va="top",
        fontsize=14, fontweight="bold", color=TXT)
ax.text(RCX, 99, "End-to-End", ha="center", va="top",
        fontsize=14, fontweight="bold", color=TXT)

# ----- LEFT column: 5 stages -----
ly = [86, 69, 52, 35, 18]
pbox(LCX, ly[0], "Fitur Neural", BLUE_F, BLUE_E)
rbox(LCX, ly[1], "GRU", YEL_F, YEL_E)
rbox(LCX, ly[2], "Probabilitas\nFonem", BLUE_F, BLUE_E)
rbox(LCX, ly[3], "Model Bahasa\n(+ rescore)", YEL_F, YEL_E)
pbox(LCX, ly[4], "Prediksi Teks", BLUE_F, BLUE_E)
for a, b in zip(ly[:-1], ly[1:]):
    arrow(LCX, a - BH/2, b + BH/2)

# ----- error-propagation brace + label (spans BiRNN -> Model Bahasa) -----
bx = LCX + BW/2 + 1.5
top_y, bot_y = ly[1] + BH/2, ly[3] - BH/2
verts = [(bx, top_y), (bx+2.5, top_y), (bx+2.5, (top_y+bot_y)/2+1.2),
         (bx+5, (top_y+bot_y)/2), (bx+2.5, (top_y+bot_y)/2-1.2),
         (bx+2.5, bot_y), (bx, bot_y)]
codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO,
         Path.LINETO, Path.LINETO, Path.LINETO]
ax.add_patch(PathPatch(Path(verts, codes), fill=False,
                       edgecolor=EP_RED, linewidth=1.6))
ax.text(bx+6, (top_y+bot_y)/2, "error\npropagation", ha="left", va="center",
        fontsize=14, style="italic", color=EP_RED, rotation=0)

# ----- RIGHT column: 3 stages (aligned top & bottom with left) -----
ry = [86, 52, 18]
pbox(RCX, ry[0], "Fitur Neural", BLUE_F, BLUE_E)
rbox(RCX, ry[1], "Model\nEnd-to-End", RED_F, RED_E)
pbox(RCX, ry[2], "Prediksi Teks", BLUE_F, BLUE_E)
for a, b in zip(ry[:-1], ry[1:]):
    arrow(RCX, a - BH/2, b + BH/2)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = "slide6_pipeline.png"
fig.savefig(out, transparent=True, bbox_inches="tight", pad_inches=0.05)
print("saved", out)
