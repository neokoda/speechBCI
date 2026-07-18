"""Diagram proses ensembling (regresi logistik dua fitur) untuk Gambar IV.2.

Dua sistem terbaik (A = dua tahap Conformer+5-gram+LLaMA-2 7B, B = E2E
Whisper-large-v3) masing-masing menghasilkan satu hipotesis beserta satu skor
keyakinan. Kedua skor keyakinan menjadi masukan sebuah model regresi logistik
yang mengeluarkan peluang p. Bila p > 0,5 dipilih hipotesis E2E, bila tidak
dipilih hipotesis dua tahap.

Tata letak vertikal (alur dari atas ke bawah). Gaya (warna, rounded box,
arrow) mengikuti make_dua_tahap_horizontal.py.
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

MAIN = 20
H = 0.62          # tinggi kotak satu baris (inci)
PAD = 0.28        # padding teks horizontal per sisi (inci)
HGAP = 0.9        # jarak horizontal antar dua cabang (inci)
STEP = 1.30       # jarak vertikal antar baris (inci)

fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
fig.canvas.draw()
REND = fig.canvas.get_renderer()


def text_w(s):
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


# ── ukuran kolom kiri (A) dan kanan (B) ─────────────────────────────────────
a_sys, a_skor = "A: dua tahap", "Keyakinan A"
b_sys, b_skor = "B: E2E", "Keyakinan B"
wL = max(text_w(a_sys), text_w(a_skor)) + 2 * PAD
wR = max(text_w(b_sys), text_w(b_skor)) + 2 * PAD
xL = wL / 2
xR = xL + wL / 2 + HGAP + wR / 2
xc = (xL + xR) / 2

# ── kotak rantai tengah ─────────────────────────────────────────────────────
lr_txt, pilih_txt, teks_txt = "Regresi logistik", "Pilih hipotesis (E2E bila p > 0,5)", "Teks akhir"
w_lr = text_w(lr_txt) + 2 * PAD
w_pilih = text_w(pilih_txt) + 2 * PAD
w_teks = text_w(teks_txt) + 2 * PAD

y0, y1, y2, y3, y4 = 0, -STEP, -2 * STEP, -3 * STEP, -4 * STEP

# baris atas: dua sistem
box(xL, y0, wL, a_sys, YEL_F, YEL_E)
box(xR, y0, wR, b_sys, BLUE_F, BLUE_E)
# baris kedua: dua skor keyakinan
box(xL, y1, wL, a_skor, GRY_F, GRY_E)
box(xR, y1, wR, b_skor, GRY_F, GRY_E)
# rantai tengah
box(xc, y2, w_lr, lr_txt, GRN_F, GRN_E)
box(xc, y3, w_pilih, pilih_txt, YEL_F, YEL_E)
box(xc, y4, w_teks, teks_txt, GRN_F, GRN_E)

# ── panah (alur ke bawah) ───────────────────────────────────────────────────
arrow((xL, y0 - H / 2), (xL, y1 + H / 2))
arrow((xR, y0 - H / 2), (xR, y1 + H / 2))
arrow((xL, y1 - H / 2), (xc - 0.15, y2 + H / 2))
arrow((xR, y1 - H / 2), (xc + 0.15, y2 + H / 2))
arrow((xc, y2 - H / 2), (xc, y3 + H / 2))
arrow((xc, y3 - H / 2), (xc, y4 + H / 2))

# ── batas gambar ────────────────────────────────────────────────────────────
xmin = min(xL - wL / 2, xc - w_pilih / 2) - 0.4
xmax = max(xR + wR / 2, xc + w_pilih / 2) + 0.4
ymin, ymax = y4 - H / 2 - 0.4, y0 + H / 2 + 0.4
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.axis("off")
ax.set_position([0, 0, 1, 1])
fig.set_size_inches(xmax - xmin, ymax - ymin)
out = os.path.join(HERE, "fig_ensemble.png")
fig.savefig(out, dpi=200, facecolor="white")
plt.close(fig)
print("wrote", out)
