"""Generate Bab III figures (matplotlib, no Graphviz needed).

Run:  python docs/thesis/figures/make_figures.py
Outputs PNGs next to this file.

Design goal: figures are COMPACT (small canvas, wrapped labels) so the text is
large relative to the boxes. When embedded at page width, the labels stay
legible. Lines prefixed with "@i@" render italic (English technical terms).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))

BLUE_F, BLUE_E = "#e8f0fe", "#1a73e8"
YEL_F,  YEL_E  = "#fef7e0", "#f9ab00"
RED_F,  RED_E  = "#fce8e6", "#d93025"
GRN_F,  GRN_E  = "#e6f4ea", "#188038"
GRY_F,  GRY_E  = "#f1f3f4", "#5f6368"

MAIN = 13      # main box text
SUB  = 11.5    # secondary / italic detail lines
TITLE = 14
SMALL = 11
LINE_H = 0.42  # vertical spacing between lines inside a box (data units)


def box(ax, x, y, w, h, text, fc=BLUE_F, ec=BLUE_E, fs=MAIN):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.01,rounding_size=0.06",
                       linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    lines = text.split("\n")
    n = len(lines)
    y0 = y + (n - 1) * LINE_H / 2
    for i, ln in enumerate(lines):
        italic = ln.startswith("@i@")
        ln = ln[3:] if italic else ln
        ax.text(x, y0 - i * LINE_H, ln, ha="center", va="center", fontsize=fs,
                zorder=3, fontstyle="italic" if italic else "normal")


def bh(text, base=1.0):
    """Box height that grows with line count."""
    return base + 0.55 * (text.count("\n"))


def arrow(ax, p1, p2, ec="#3c4043", style="-|>", lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13,
                                 linewidth=lw, color=ec, zorder=1, linestyle=ls,
                                 shrinkA=2, shrinkB=2))


def finish(fig, ax, name, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    ax.set_aspect("equal")
    fig.tight_layout(pad=0.2)
    out = os.path.join(HERE, name)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


def vstack(ax, x, steps, top=0.0, w=4.4, gap=1.55):
    """Draw a vertical stack of (text, fc, ec) boxes; return list of (y, h)."""
    ys, y = [], top
    for txt, fc, ec in steps:
        h = bh(txt)
        box(ax, x, y - h / 2, w, h, txt, fc=fc, ec=ec)
        ys.append((y - h / 2, h))
        y -= h + gap
    for i in range(len(steps) - 1):
        (cy0, h0), (cy1, h1) = ys[i], ys[i + 1]
        arrow(ax, (x, cy0 - h0 / 2), (x, cy1 + h1 / 2))
    return ys


# ---------------------------------------------------------------------------
# Gambar III.2 — Praproses dan ekstraksi fitur
# ---------------------------------------------------------------------------
def fig_preprocess():
    fig, ax = plt.subplots(figsize=(5.0, 9.2))
    W, x = 5, 0
    steps = [
        ("Sinyal ECoG mentah\n128 kanal elektroda", GRY_F, GRY_E),
        ("Fitur per bin 20 ms\n@i@128 threshold crossings\n@i@+ 128 spike band power", BLUE_F, BLUE_E),
        ("Matriks fitur  T x 256", BLUE_F, BLUE_E),
        ("Normalisasi z-score\nper sesi", BLUE_F, BLUE_E),
        ("Gaussian smoothing +\naugmentasi (saat latih)\n@i@white noise, constant offset", GRN_F, GRN_E),
        ("Input ke model\n(dua tahap & end-to-end)", YEL_F, YEL_E),
    ]
    ys = vstack(ax, x, steps, w=W, gap=1.1)
    last_y, last_h = ys[-1]
    finish(fig, ax, "fig_praproses.png",
           (-W / 2 - 0.3, W / 2 + 0.3), (last_y - last_h, 0.9))


# ---------------------------------------------------------------------------
# Gambar III.3 — Arsitektur dua tahap
# ---------------------------------------------------------------------------
def fig_two_stage():
    fig, ax = plt.subplots(figsize=(5.6, 10.0))
    W, x = 4.6, 0
    steps = [
        ("Fitur neural  T x 256", GRY_F, GRY_E),
        ("Model berbasis\nTransformer\n(dekoder fonem)", YEL_F, YEL_E),
        ("Probabilitas fonem (CTC)\n40 kelas + token kosong", YEL_F, YEL_E),
        ("Dekode WFST 5-gram\n@i@beam search +\n@i@shallow fusion", BLUE_F, BLUE_E),
        ("Daftar n-best hipotesis", BLUE_F, BLUE_E),
        ("Rescoring LLaMA-2 7B", BLUE_F, BLUE_E),
        ("Teks akhir", GRN_F, GRN_E),
    ]
    ys = vstack(ax, x, steps, w=W, gap=1.05)
    # stage brackets
    bx = W / 2 + 0.35
    y_top1, h_top1 = ys[1]
    y_bot1, h_bot1 = ys[2]
    ax.annotate("", xy=(bx, y_top1 + h_top1 / 2), xytext=(bx, y_bot1 - h_bot1 / 2),
                arrowprops=dict(arrowstyle="-", color=YEL_E, lw=2))
    ax.text(bx + 0.12, (y_top1 + y_bot1) / 2, "Tahap 1\n(model akustik)",
            ha="left", va="center", fontsize=SMALL, color=YEL_E)
    y_top2, h_top2 = ys[3]
    y_bot2, h_bot2 = ys[5]
    ax.annotate("", xy=(bx, y_top2 + h_top2 / 2), xytext=(bx, y_bot2 - h_bot2 / 2),
                arrowprops=dict(arrowstyle="-", color=BLUE_E, lw=2))
    ax.text(bx + 0.12, (y_top2 + y_bot2) / 2, "Tahap 2\n(model bahasa)",
            ha="left", va="center", fontsize=SMALL, color=BLUE_E)
    last_y, last_h = ys[-1]
    finish(fig, ax, "fig_dua_tahap.png",
           (-W / 2 - 0.3, W / 2 + 2.1), (last_y - last_h, 0.9))


# ---------------------------------------------------------------------------
# Gambar III.5 — Arsitektur end-to-end (cross-attention)
# ---------------------------------------------------------------------------
def fig_e2e():
    fig, ax = plt.subplots(figsize=(7.6, 7.4))
    Wl = 4.25
    xl, xr = -2.55, 2.55
    gap = 1.2

    left = [
        ("ECoG\n(B, T, 256)", GRY_F, GRY_E),
        ("Conformer encoder\n@i@d=512, spatial attn,\n@i@subsample 4x", RED_F, RED_E),
        ("Proyektor\nLinear 512 -> d_FM\n+ LayerNorm", RED_F, RED_E),
        ("ECoG memory\n(B, T', d_FM)", GRN_F, GRN_E),
    ]
    yl = vstack(ax, xl, left, w=Wl, gap=gap)

    right = [
        ("Token teks\n(teacher forcing /\nautoregresif)", BLUE_F, BLUE_E),
        ("@i@Self-attention\n(teks -> teks)", BLUE_F, BLUE_E),
        ("@i@Cross-attention\nQ = teks,\nK = V = ECoG memory", BLUE_F, BLUE_E),
        ("@i@Feed-forward (FFN)", BLUE_F, BLUE_E),
    ]
    yr = vstack(ax, xr, right, w=Wl, gap=gap)

    # bracket: x N lapisan decoder FM (LoRA)
    bx = xr + Wl / 2 + 0.25
    yt, ht = yr[1]
    ybt, hbt = yr[3]
    ax.annotate("", xy=(bx, yt + ht / 2), xytext=(bx, ybt - hbt / 2),
                arrowprops=dict(arrowstyle="-", color=BLUE_E, lw=2))
    ax.text(bx + 0.12, (yt + ybt) / 2, "x N lapisan\ndecoder FM\n(LoRA)",
            ha="left", va="center", fontsize=SMALL, color=BLUE_E)

    # cross link ECoG memory -> cross-attention
    ly, lh = yl[3]
    ry, rh = yr[2]
    arrow(ax, (xl + Wl / 2, ly), (xr - Wl / 2, ry), ec=GRN_E, lw=1.8, ls=(0, (4, 2)))
    ax.text(0, (ly + ry) / 2 + 1.5, "K, V", ha="center", va="center",
            fontsize=SUB, color=GRN_E)

    # output below right column
    ylast, hlast = yr[-1]
    y_out = ylast - hlast / 2 - gap - 0.5
    box(ax, xr, y_out, Wl, 1.0, "LM head -> teks akhir", fc=GRN_F, ec=GRN_E)
    arrow(ax, (xr, ylast - hlast / 2), (xr, y_out + 0.5))

    finish(fig, ax, "fig_e2e.png",
           (xl - Wl / 2 - 0.4, bx + 1.9), (y_out - 0.6, 0.9))


# ---------------------------------------------------------------------------
# Gambar III.4 — Arsitektur end-to-end gaya LLaVA (decoder-only)
# ---------------------------------------------------------------------------
def fig_e2e_llava():
    fig, ax = plt.subplots(figsize=(7.2, 8.0))
    Wl = 5
    xl, xr = -2.75, 2.75
    gap = 1.15

    left = [
        ("ECoG\n(B, T, 256)", GRY_F, GRY_E),
        ("Conformer encoder\n@i@d=512, spatial attn,\n@i@subsample 4x", RED_F, RED_E),
        ("Proyektor\nLinear 512 -> d_LM\n+ LayerNorm", RED_F, RED_E),
        ("Token ECoG\n(B, T', d_LM)", GRN_F, GRN_E),
    ]
    yl = vstack(ax, xl, left, w=Wl, gap=gap)

    # Text token box (right, level with Token ECoG)
    ly3, lh3 = yl[3]
    box(ax, xr, ly3, Wl, lh3, "Token teks", fc=BLUE_F, ec=BLUE_E)

    # concat
    y_cat = ly3 - lh3 / 2 - gap - 0.55
    box(ax, 0, y_cat, 6, 1.5, "Konkatenasi\n[ Token ECoG | Token teks ]", fc=BLUE_F, ec=BLUE_E)
    arrow(ax, (xl, ly3 - lh3 / 2), (-1.0, y_cat + 0.55))
    arrow(ax, (xr, ly3 - lh3 / 2), (1.0, y_cat + 0.55))

    y_lm = y_cat - 0.55 - gap - 0.5
    box(ax, 0, y_lm, 5.4, 1.0, "Decoder-only LM (LoRA)", fc=BLUE_F, ec=BLUE_E)
    arrow(ax, (0, y_cat - 0.55), (0, y_lm + 0.5))

    y_out = y_lm - 0.5 - gap - 0.5
    box(ax, 0, y_out, 5.4, 1.0, "LM head -> teks akhir", fc=GRN_F, ec=GRN_E)
    arrow(ax, (0, y_lm - 0.5), (0, y_out + 0.5))

    finish(fig, ax, "fig_e2e_llava.png",
           (xl - Wl / 2 - 0.4, xr + Wl / 2 + 0.4), (y_out - 0.6, 0.9))


# ---------------------------------------------------------------------------
# Gambar III.1 — Perbandingan blok Transformer vs Conformer vs +spatial
# ---------------------------------------------------------------------------
def fig_variants():
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    W, H = 3.5, 1
    gap = 0.5
    MHSA = "@i@Multi-Head\n@i@Self-Attention *"
    FF   = "@i@Feed-Forward *"
    FFh  = "@i@Feed-Forward\n@i@(1/2) *"
    cols = {
        "Transformer": (-4.0, [
            (MHSA, BLUE_F, BLUE_E),
            ("@i@Add & Norm", GRY_F, GRY_E),
            (FF, BLUE_F, BLUE_E),
            ("@i@Add & Norm", GRY_F, GRY_E),
        ]),
        "Conformer": (0.0, [
            (FFh, BLUE_F, BLUE_E),
            (MHSA, BLUE_F, BLUE_E),
            ("@i@Convolution\n@i@Module", YEL_F, YEL_E),
            (FFh, BLUE_F, BLUE_E),
            ("@i@LayerNorm", GRY_F, GRY_E),
        ]),
        "Conformer +\nSpatial Attention": (4.0, [
            ("@i@Spatial Attention\n(antarelektroda)", RED_F, RED_E),
            (FFh, BLUE_F, BLUE_E),
            (MHSA, BLUE_F, BLUE_E),
            ("@i@Convolution\n@i@Module", YEL_F, YEL_E),
            (FFh, BLUE_F, BLUE_E),
            ("@i@LayerNorm", GRY_F, GRY_E),
        ]),
    }
    top = 0.0
    bottoms = []
    for title, (cx, steps) in cols.items():
        ax.text(cx, top + 1.5, title, ha="center", va="center",
                fontsize=TITLE, fontweight="bold")
        y = top
        ys = []
        for txt, fc, ec in steps:
            h = bh(txt, base=0.95)
            box(ax, cx, y - h / 2, W, h, txt, fc=fc, ec=ec, fs=SUB)
            ys.append((y - h / 2, h))
            y -= h + gap
        for i in range(len(steps) - 1):
            (cy0, h0), (cy1, h1) = ys[i], ys[i + 1]
            arrow(ax, (cx, cy0 - h0 / 2), (cx, cy1 + h1 / 2))
        bottoms.append(ys[-1][0] - ys[-1][1])
    yb = min(bottoms) - 0.6
    legend = ("* hiperparameter yang disetel: jumlah head (MHSA), dimensi feed-forward,\n"
              "jumlah lapisan. Selain itu, d_model (seluruh blok), learning rate, dan\n"
              "jadwal peluruhannya juga disetel.")
    ax.text(0, yb, legend, ha="center", va="top", fontsize=SMALL, color="#3c4043")
    finish(fig, ax, "fig_varian_transformer.png",
           (-6.0, 6.0), (yb - 1.0, top + 2.2))


if __name__ == "__main__":
    # fig_preprocess()
    # fig_two_stage()
    # fig_e2e()
    # fig_e2e_llava()
    fig_variants()
    print("done")
