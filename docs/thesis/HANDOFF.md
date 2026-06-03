# Thesis Writing Handoff

**Scope:** progress + context for *writing the thesis document* (Bahasa Indonesia). This is separate from the repo-root `HANDOFF.md`, which tracks *experiments*. When you need result numbers, the root `HANDOFF.md` and `docs/mds/EXPERIMENTS.md` are the source of truth.

**Last updated:** 2026-06-02 (**SEMUA BAB I–V SUDAH DIDRAFT — penulisan tugas akhir dijeda untuk sekarang. Tidak ada bab yang sedang ditulis aktif.** Sesi ini finalisasi Bab II: +Gambar II.4 `conformer.png` (Gulati 2020) dgn penjelasan blok (SpecAugment, convolution subsampling 10→40 ms ≈ 4 langkah masukan per langkah keluaran, residual setengah, layer normalization [Ba 2016], kernel 31), +Gambar II.5 `lora.png` (Hu 2021), GRU dijelaskan vs RNN [Cho 2014], rescoring neural diberi rumus, contoh konkret 'i am better' (fitur→fonem→teks) + trellis beam search + Tabel II.1 n-best/rescoring di II.3.3. **Spatial attention dipindah ke II.4.1 & dibingkai 'dapat dicoba karena berpotensi' (BUKAN dipakai); II.4.3 lama DIHAPUS.** Dafpus dibersihkan: +Cho/Ba/Park (web-verified), 8 orphan tak-tersitasi dihapus (Bai, Card, Gemma, Li, Luo, Oord, Rainey, Willett 2025), Touvron dipastikan LLaMA-2 (2307.09288), Silva dilengkapi (NRN 25(7):473-492) → kini 34 entri = persis set yang disitasi di bab1–5. | Sebelumnya 2026-06-01: Bab II didraft ke `bab2.md` — 6 bagian, 3 gambar baru, lihat §4c. Dataset naik jadi II.2, II.4.4 lama dihapus, dafpus +Mohri 2002 +Qwen3. | Sebelumnya: 2026-05-29 — Bab IV drafted ke `bab4.md` — IV.1 Lingkungan, IV.2 Desain, IV.3 Hasil, IV.4 Pembahasan. Eksperimen dekode fonem kini 4 dekoder (+ Transformer murni PER 0,2444 @4_18). Evaluasi arsitektur hanya pelaporkan irisan willett_4_18 (24-sess dibuang). Dua tahap kini menampilkan GRU/Transformer/Conformer. IV.4 diperkaya dgn error analysis (CPU-only, dari cache eval_full.json + _nbest_tmp.json): angka entropi 0,0950 vs 0,1448 nats di paragraf 'ketajaman distribusi', plus 4 paragraf baru — komposisi Ins/Del/Sub (E2E v7 vs LM2, didominasi Sub ~95%), WER vs panjang ujaran + Gambar IV.1 (`figures/fig_wer_vs_length.png`), oracle WER + coverage (willett_4_18 coverage 68,3% / no-cov 31,7% / oracle WER 0,0758 vs LM6 0,1556), dan overlap E2E vs LM2 (24,2% komplementer). Skrip: `AnalysisExamples/analyze_errors.py`. Hasil JSON: `experiments/analysis/error_analysis.json`. Catatan: angka "46,9% coverage" lama dari HANDOFF root §5 tidak match dgn cache n-best yang sekarang (beam24/nb200) — pakai angka aktual hasil pengukuran ini. Placeholder tersisa: Prosesor/RAM pod, semua kolom Transformer dua tahap, CER Transformer, Canary/Granite WER+CER. Next: ambil GPU utk Bagian B (data sweep) + LM6 per-utt rescoring kalau mau Ins/Del/Sub LM6.)

---

## 1. Files in this folder

| File                               | Role                                                                                                         |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `bab1.md`                        | BAB I PENDAHULUAN — drafted, under active revision by user + assistant.                                     |
| `bab2.md`                        | BAB II KAJIAN PUSTAKA — **draft selesai** (lihat §4c).                                                      |
| `bab3.md`                        | BAB III ANALISIS MASALAH DAN RANCANGAN SOLUSI — **draft selesai** (lihat §4b).                              |
| `figures/`                       | Gambar Bab III (PNG) + `make_figures.py` (generator matplotlib). User sudah merapikan.                      |
| `WRITING.md`                     | Writing rules.**Read before writing/editing any prose.**                                               |
| `EYD.md`                         | Ejaan/punctuation rules. Referenced by WRITING.md.                                                           |
| `DAFTAR_PUSTAKA.md`              | Bibliography. Add a reference only after web-verifying all details.                                          |
| `template-laporanTugasAkhir.pdf` | Official ITB IF template. Defines Bab structure (note: requires §1.6 Sistematika Pembahasan).               |
| (refs in `../refs/`)             | Willett 2023 (`s41586-023-06377-x.pdf`), Seto thesis (`laporanTugasAkhir-13521081-FINALFINAL.docx.pdf`). |

PDFs on this Windows machine: `pdftoppm` is NOT installed, so the Read tool fails on PDFs. Use PyMuPDF instead: `python -c "import fitz; ..."` with `$env:PYTHONIOENCODING="utf-8"` (the template/Seto thesis were read this way).

---

## 2. Writing rules (summary — full text in WRITING.md / EYD.md)

1. Diksi sederhana — pilih kata paling sederhana yang tidak mengubah makna.
2. Selalu sitasi, dan **verifikasi via web sebelum menambah** ke `DAFTAR_PUSTAKA.md` (penulis, judul, tahun, venue, volume, halaman). Jangan pernah menulis detail sitasi karangan.
3. Hindari titik koma `;`, titik dua `:` di dalam kalimat, dan em dash `—`. (Titik koma di dalam kurung sitasi berganda tetap boleh — itu konvensi.)
4. EYD: jangan dahului "sehingga"/"dengan" dengan koma; pakai "Oleh karena itu," bukan "Karena itu,".

---

## 3. Chapter status

**Seluruh bab (I–V) sudah didraft. Penulisan tugas akhir dijeda untuk sekarang — tidak ada bab yang sedang ditulis aktif. Pekerjaan selanjutnya tinggal revisi/poles, mengisi placeholder angka (spek pod, Transformer dua tahap, Canary/Granite, RTF/WPM), dan sinkronisasi daftar pustaka ke dokumen Word.**

| Bab                          | Status                        | Catatan                                                  |
| ---------------------------- | ----------------------------- | -------------------------------------------------------- |
| I Pendahuluan                | **Draft selesai** | §1.1–§1.6 ada. Lihat keputusan framing di §4.        |
| II Kajian Pustaka            | **Draft selesai (5 gambar)** | `bab2.md`. Lihat §4c. Feedback dosen #2 (dataset) terpenuhi via II.2. Spatial attention kini di II.4.1 (II.4.3 dihapus). |
| III Analisis & Rancangan     | **Draft selesai (5 gambar)**  | Lihat §4b untuk struktur + keputusan/koreksi.            |
| IV Evaluasi                  | **Draft selesai (struktur 5 bagian)** | `bab4.md`. IV.1 Lingkungan, IV.2 Desain, IV.3 Hasil, IV.4 Analisis (3 subbagian: IV.4.1 entropi+coverage/oracle GRU vs Conformer dari cache asc=0.5 yg konsisten dgn Tabel IV.3, IV.4.2 WER vs panjang ujaran (E2E unggul di hampir semua bucket kecuali 5 kata), IV.4.3 komplementaritas+best-of-two oracle WER 0,1249), IV.5 Pembahasan. Aturan #5 WRITING.md ttg hindari pola partisipial English-style. Skrip `AnalysisExamples/analyze_errors.py` updated utk pakai cache asc=0.5. Placeholder: spek pod, Transformer dua tahap, Canary/Granite WER+CER. |
| V Kesimpulan & Saran         | **Draft selesai**             | `bab5.md`. V.1 Kesimpulan (3 poin selaras §1.3: dua-tahap menang, E2E kompetitif belum menang, analisis lengkap). V.2 Saran (5 poin: data+pralatih, rescorer lebih baik, ensembling, speaker-independent, perluasan kosakata). |

---

## 4. Keputusan framing Bab I (jaga konsistensi di bab lain)

- **Payung "berbasis Transformer".** Conformer disebut sebagai *varian* Transformer untuk domain ucapan, bukan paradigma terpisah. Menghindari "missing link" RNN→Conformer.
- **Speaker dependency** ada di **Batasan** (poin 1), bukan Latar Belakang. (Awalnya feedback dosen, tapi user memutuskan taruh di Batasan.)
- **Urutan argumen E2E** di §1.1: error propagation (pendukung) → dua keunggulan struktural (joint optimization + pengetahuan FM dipakai langsung dalam dekode, bukan rescoring terpisah) → kenapa FM (data terbatas) → skalabilitas/ekstensibilitas. **Jangan** menjadikan "error propagation hilang" sebagai klaim utama — secara empiris dua tahap masih menang, jadi itu akan dibantah hasil sendiri.
- **Latar Belakang = cerita masalah.** Detail implementasi (arsitektur konkret, slice, dsb.) ditunda ke Bab II/III. User eksplisit soal ini.
- **Related work (Feng 2024, Zhang/BIT 2025) ditunda ke Bab II §II.5.** Sengaja tidak di Latar Belakang.

## 4b. Bab III — struktur + keputusan/koreksi (jaga konsistensi di Bab IV)

`bab3.md` draft selesai. Struktur:

- **III.1 Analisis Masalah** (4 masalah): (1) Pemilihan Varian Transformer *(baru)*, (2) Pemilihan Arsitektur FM, (3) Adaptasi Model terhadap Input Baru, (4) Latensi Inferensi & Beban Komputasi.
- **III.2 Analisis Solusi** (paralel dengan III.1; memuat **Tabel III.1** daftar FM).
- **III.3 Rancangan Solusi**: III.3.1 Praproses & Ekstraksi Fitur (bersama dua arsitektur), III.3.2 Dua Tahap (Dekoder Fonem / Pelatihan / Model Bahasa), III.3.3 E2E (Penyiapan / Pelatihan).

Gambar (`figures/`, generator `make_figures.py`, **sudah dirapikan user**): III.1 `fig_varian_transformer`, III.2 `fig_praproses`, III.3 `fig_dua_tahap`, III.4 `fig_e2e_llava`, III.5 `fig_e2e`.

**Keputusan & koreksi penting (PAKAI JUGA DI BAB IV):**

- **Hasil sengaja TIDAK ada di Bab III** — semua angka PER/WER ditunda ke Bab IV.
- **Tabel III.1 (6 FM):** Whisper-medium.en = **769 juta** (bukan 244 jt). **Canary-Qwen-2.5B & Granite-Speech-4.1-2B dipakai gaya *LLaVA*** (pakai ulang LM teks di dalamnya), **BUKAN cross-attention**. Hanya **Whisper & Cohere = cross-attention**. (Canary: pakai ulang Qwen3-1.7B + proyektor/LoRA Canary; Granite: hanya `language_model`-nya, encoder audio dibuang.)
- **Tidak ada penyaringan *high-gamma*/*low-frequency*** di pipeline kita — fitur Willett = *threshold crossings* + *spike band power* (pita spike). Klaim *high-gamma* sudah **dihapus** dari Bab I (metodologi) & Bab III.
- Fitur **256 = 128 kanal × 2** (128 *tx* + 128 *spikePow*). Implan = **2×96 Utah array**, 128 kanal (area 6v) dipakai. Penamaan **"ECoG" tetap dipertahankan** (keputusan sengaja walau sebenarnya intrakortikal).
- **Encoder E2E di-pretrain dengan CTC level-karakter** (`AnalysisExamples/e2e/train_ctc.py`), TERPISAH dari dekoder fonem dua tahap (yang fonem-level, TensorFlow). Saat E2E, **encoder + proyektor tidak dibekukan** (ikut di-fine-tune, LR kecil); dekoder FM hanya di-LoRA.
- Konfig dekoder fonem dua tahap (Conformer-spatial): batch 32, LR 0.04→0.004 *cosine*, *warmup* 1000, 150k langkah, *early stop* patience 50, best PER di langkah 126000.
- Konfig E2E headline (v7 Whisper-large-v3): LoRA r=16/α=32/dropout=0,1 pada attn + cross-attn + FFN; 15k langkah; batch efektif 16; LR encoder 6,9e-5 / proyektor 1e-3 / LoRA 1,75e-4.
- **Dafpus:** ditambah **Simeral et al. (2021, IEEE TBME 68(7))** + **LLaMA-2 (Touvron et al., 2023; arXiv:2307.09288)** — entri LLaMA-1 lama **dihapus** (LLaMA-2 jadi "Touvron et al., 2023" biasa). **Cohere/Canary/Granite TIDAK** ditambah sebagai sitasi formal (tak ada paper; hanya nama tool). Sitasi *speaker-dependency* (Willett 2023; Silva 2024) ditambah ke **Bab I §1.4**.

---

## 4c. Bab II — struktur + keputusan (jaga konsistensi)

`bab2.md` draft selesai. Struktur final (penomoran berubah dari proposal karena Dataset naik jadi II.2):

- **II.1 BCI** (II.1.1 Paradigma, II.1.2 Akuisisi, II.1.3 ECoG, **II.1.4 "Produksi Ucapan dalam Otak"** — judul diganti dari "Mekanisme Neural...").
- **II.2 Dataset Willett et al. (2023)** *(BARU, feedback dosen #2)* — T12/ALS, 24 sesi, tx1 (-3.5xRMS) + spikePow (high-pass 250 Hz, µV²), 128 kanal area 6v → 256-dim, Switchboard+OpenWebText (sumber: readme dataset baris 5, **bukan halusinasi Seto**), 8800 latih/880 uji.
- **II.3 Pemrosesan Sinyal ECoG → Teks** (II.3.1 Praproses, II.3.2 Fonem+CTC, II.3.3 Fonem→Teks dgn **WFST/beam search/shallow fusion/n-best rescoring**, II.3.4 Metrik).
- **II.4 Model Berbasis Transformer** (II.4.1 Transformer + **spatial attention** sebagai penerapan *attention* pada dimensi elektroda yang dibingkai "dapat dicoba karena berpotensi", II.4.2 Conformer + **Gambar II.4 `conformer.png`** Gulati 2020). *II.4.3 Spatial Attention subbab terpisah sudah DIHAPUS — dilebur ke II.4.1.*
- **II.5 Foundation Model** (II.5.1 Varian + Tabel II.2 daftar FM aktual, II.5.2 Adaptasi modalitas = **3 teknik**: proyeksi linier/LLaVA/cross-attention, II.5.3 **PEFT dgn LoRA** + rumus). *II.4.4 Optimasi Kecepatan lama DIHAPUS.*
- **II.6 Penelitian Terkait** (Willett, Seto, LaBraM/CBraMod, +Feng 2024, +Zhang 2025).

**Keputusan & koreksi (selaras Bab III §4b):**
- Gambar BARU: II.1 `fig_dataset`, II.2 `fig_ecog_pipeline`, II.3 `fig_transformer_variants` (generator terpisah `figures/make_bab2_figures.py`, reuse helper dari `make_figures.py` — TIDAK menyentuh gambar Bab III); II.4 `conformer.png` (Gulati 2020, ditaruh user), II.5 `lora.png` (Hu 2021, ditaruh user). Total **5 gambar Bab II**.
- Rumus ditambahkan: CTC, self-attention, shallow fusion, **rescoring neural (s_ak·log p_akustik + α·log p_neural + β·N_kata)**, LoRA (W=W₀+(α/r)BA).
- **Koreksi info salah proposal**: hapus klaim *high-gamma*/band-pass 70-200 Hz (pipeline pakai threshold crossings + spike band power); jumlah kelas fonem = 39+jeda+blank (T×41); model bahasa = 5-gram WFST + rescoring LLaMA-2 (bukan "Transformer").
- **Dafpus ditambah**: Mohri et al. (2002, Computer Speech & Language 16(1):69-88) untuk WFST; Qwen Team (2025, arXiv:2505.09388) untuk Qwen3. Cohere/Canary/Granite TETAP tanpa sitasi formal. **Sesi 2026-06-02:** +Cho 2014 (GRU), +Ba 2016 (LayerNorm), +Park 2019 (SpecAugment) — semua web-verified; **8 orphan tak-tersitasi DIHAPUS** (Bai, Card, Gemma, Li, Luo, Oord, Rainey, Willett 2025); Touvron dipastikan **LLaMA-2** (2307.09288); Silva 2024 dilengkapi (NRN 25(7):473-492). Dafpus final = **34 entri**, persis sama dengan himpunan sitasi di bab1–5 (sudah dicek silang otomatis).
- **bab1 §1.6** sudah diperbarui agar urutan Bab II mencerminkan Dataset sebagai II.2.
- Placeholder/TODO: belum ada gambar arsitektur Transformer asli (Vaswani) — cukup pakai perbandingan blok di Gambar II.3.

## 5. Feedback dosen (checklist — dari `../proposal/feedback.txt`)

1. Jelaskan masalah *speaker dependency* di riset BCI. → **Sudah** di Batasan Bab I; bisa diperdalam di Bab II.
2. Tambah bagian di Bab II tentang deskripsi dataset + praproses. → **Sudah** di Bab II §II.2 (dataset) + §II.3.1 (praproses).
3. Jelaskan detail bagaimana sinyal diadaptasi ke input model. → **Sudah** di Bab II §II.5.2 (3 teknik adaptasi modalitas) + Bab III §III.3.3.

---

## 6. Rencana Bab II (saat mulai)

Ikuti struktur proposal §II + template, plus dua item feedback dosen. Urutan yang disarankan:

- II.1 Brain-Computer Interface (paradigma, akuisisi sinyal, ECoG, mekanisme neural produksi ucapan).
- II.2 Pemrosesan sinyal ECoG menjadi teks (praproses + ekstraksi fitur, dekode fonem, fonem→teks, metrik).
- II.3 Transformer **dan Conformer** (Conformer dapat subbab sendiri di sini — penjelasan modul konvolusi yang ditunda dari Bab I).
- II.4 Foundation Model (varian arsitektur, adaptasi modalitas, fine-tuning/PEFT, optimasi kecepatan).
- II.5 Penelitian terkait — **Willett 2023, Seto 2025, Feng 2024 (BrainLLM, gaya LLaVA), Zhang/BIT 2025 (encoder lintas-spesies + audio LLM + contrastive)**. Posisikan kontribusi tesis sebagai studi perbandingan empiris beberapa paradigma adaptasi FM di rezim satu-lab (tanpa pretraining lintas-spesies skala besar). Lihat plan di `~/.claude/plans/can-you-understand-this-floating-pebble.md` §1.1.d untuk uraian lengkap kedua paper.
- II.6 Dataset (deskripsi Willett 2023 + praproses) — **memenuhi feedback #2.**

Materi proposal lama (`../proposal/proposal.txt`) bisa dipakai ulang untuk II.1–II.4 (sudah ditulis di proposal).

---

## 7. Angka hasil untuk dikutip (terverifikasi; sumber: root HANDOFF.md / EXPERIMENTS.md)

Baseline:

- Willett et al. (2023): WER 17,4%.
- Seto et al. (2025): WER 16,9%, PER 19,2%. (Slice yang dilaporkan ~ willett_4_18. Catatan: laporan Seto keliru menyebut data Metzger; user konfirmasi datanya Willett T12.)

Hasil tugas akhir ini (slice: willett_4_18 / all_24):

- Dekoder fonem Conformer-spatial (P6): PER **0,1428** (willett_4_18) / 0,1654 (all_24). Mengalahkan GRU Seto (PER ~0,192).
- Dua tahap terbaik (LM6 = Conformer + 5-gram + base LLaMA-2 7B rescore): WER **0,1556** (willett_4_18) / 0,1897 (all_24); CER 0,1127 (willett_4_18).
- E2E terbaik (v7 = Whisper-large-v3 cross-attention): WER **0,1716** (willett_4_18) / 0,2053 (all_24).
- E2E alternatif (Cohere v3-ext3): WER 0,2254 (all_24).

Pembanding SOTA (Zhang/BIT 2025, benchmark Brain-to-Text '24 = data Willett T12):

- E2E model tunggal 15,67%; E2E ensembel 10,22%; kaskade model tunggal 6,35%.
- Narasi jujur: v7 (0,1716) hanya ~1,5pp di belakang E2E model tunggal BIT; keunggulan BIT terutama dari pretraining lintas-spesies ~367 jam + ensembling, bukan kepintaran fine-tuning. Detail di plan file §"Why is BIT so much better".

Klaim yang aman:

- Mengalahkan Seto pada **PER** dan pada **slice willett_4_18** (WER & CER dua tahap).
- E2E **tidak** mengalahkan Seto pada WER — jangan klaim. Posisikan "kompetitif".

---

## 8. Isu data / yang perlu diingat

- **LoRA LLaMA-2 fine-tuned (LM7, WER 0,1910) hilang** dari disk & gdrive. Tidak bisa direproduksi tanpa training ulang (`finetune_llama_owt2.py`). Headline dua tahap pakai LM6 (base LLaMA, 0,1897) yang recoverable.
- **Canary & Granite (varian E2E)** tidak punya angka full-set yang valid (Canary eval rusak di transformers 5.6.2; Granite tak ada branch eval). Sebut "diimplementasikan tetapi tidak dievaluasi penuh karena inkompatibilitas pustaka", jangan masukkan tabel headline.
- **Cohere:** pakai v3-ext3 (0,2254) sebagai headline Cohere, bukan v3 (0,2394).
- **Sitasi BIT** sudah benar: Zhang, Y. et al. (2025), arXiv:2511.21740 (sebelumnya placeholder "Suresh" — sudah diperbaiki di bab1 & dafpus).

---

## 9. Referensi dokumen pendukung

- Plan/analisis lengkap (alasan framing, analisis paper, rekomendasi konferensi): `C:\Users\LENOVO\.claude\plans\can-you-understand-this-floating-pebble.md`.
- Sumber angka eksperimen: root `HANDOFF.md`, `docs/mds/EXPERIMENTS.md`.
- Proposal asli (teks lengkap untuk daur ulang Bab II): `../proposal/proposal.txt`.

---

## 10. Bab IV — titik mulai (sesi berikutnya)

Nama bab per sistematika Bab I §1.6: **"Bab IV Evaluasi"** — prosedur evaluasi, hasil tiap eksperimen, lalu pembahasan/analisis. Tulis ke `bab4.md`.

**Sumber angka (jangan karang, verifikasi):** root `HANDOFF.md` §7 + `docs/mds/EXPERIMENTS.md` (kanonik). Konvensi WER = corpus-level (micro-average), bukan rata-rata per sesi.

**Angka headline (slice willett_4_18 / all_24):**
- Dekoder fonem P6 (Conformer-spatial): PER **0,1428 / 0,1654** (kalahkan GRU Seto ~0,192).
- Dua tahap terbaik LM6 (Conformer-spatial + 5-gram + base LLaMA-2 7B): WER **0,1556 / 0,1897**; CER 0,1127 (willett_4_18).
- E2E terbaik v7 (Whisper-large-v3 cross-attn): WER **0,1716 / 0,2053**.
- E2E alternatif Cohere v3-ext3: WER 0,2254 (all_24).

**Klaim aman (lihat §7):** kalahkan Seto pada **PER** & pada **slice willett_4_18** (WER/CER dua tahap). E2E **tidak** kalahkan Seto pada WER — posisikan **"kompetitif"**, jangan klaim menang.

**Yang perlu diingat (lihat §8):**
- Canary & Granite: "diimplementasikan tetapi tidak dievaluasi penuh karena inkompatibilitas pustaka" — **jangan masukkan tabel headline**.
- LM7 (ft-LLaMA, 0,1910) hilang — headline dua tahap pakai LM6 (base LLaMA, 0,1897).
- Pembanding SOTA: Zhang/BIT 2025 (benchmark Brain-to-Text '24 = data Willett T12).
- Metrik akurasi: PER, WER, CER. Kecepatan **RTF/WPM masih TBD** (belum diimplementasi) — sebut sebagai keterbatasan/pekerjaan lanjutan, jangan tampilkan angka.

**Konsistensi dengan Bab III (§4b):** istilah & framing (ECoG, gaya LLaVA vs cross-attention, fitur tx+spikePow) harus sama. Jangan tampilkan ulang detail rancangan — Bab IV fokus prosedur + hasil + analisis.
