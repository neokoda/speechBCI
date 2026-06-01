# Thesis Writing Handoff

**Scope:** progress + context for *writing the thesis document* (Bahasa Indonesia). This is separate from the repo-root `HANDOFF.md`, which tracks *experiments*. When you need result numbers, the root `HANDOFF.md` and `docs/mds/EXPERIMENTS.md` are the source of truth.

**Last updated:** 2026-05-29 (Bab IV drafted ke `bab4.md` — IV.1 Lingkungan, IV.2 Desain, IV.3 Hasil, IV.4 Pembahasan. Eksperimen dekode fonem kini 4 dekoder (+ Transformer murni PER 0,2444 @4_18). Evaluasi arsitektur hanya pelaporkan irisan willett_4_18 (24-sess dibuang). Dua tahap kini menampilkan GRU/Transformer/Conformer. Tabel IV.4 diperluas dgn kolom params + rincian params + rincian storage + RTF + WPM dari `BENCHMARK_RESULTS.md` (Session 23, subset w4-18). WPM kini = throughput wall-clock (bukan laju referensi 63,6). Pembahasan IV.4 ditulis ulang tanpa label tebal (kalimat utuh), headline = E2E Whisper-large-v3 (paling seimbang). Placeholder tersisa: Prosesor/RAM pod, semua kolom Transformer dua tahap (tdk ada di benchmark), CER Transformer, Canary/Granite WER+CER. Next: revisi Bab IV atau mulai Bab II.)

---

## 1. Files in this folder

| File                               | Role                                                                                                         |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `bab1.md`                        | BAB I PENDAHULUAN — drafted, under active revision by user + assistant.                                     |
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

| Bab                          | Status                        | Catatan                                                  |
| ---------------------------- | ----------------------------- | -------------------------------------------------------- |
| I Pendahuluan                | **Draft, revisi aktif** | §1.1–§1.6 ada. Lihat keputusan framing di §4.        |
| II Kajian Pustaka            | Belum mulai                   | Struktur + 2 item feedback dosen wajib (lihat §5, §6). |
| III Analisis & Rancangan     | **Draft selesai (5 gambar)**  | Lihat §4b untuk struktur + keputusan/koreksi.            |
| IV Evaluasi                  | **Draft selesai**             | `bab4.md`. 2 eksperimen (dekode fonem + evaluasi arsitektur). Placeholder: spek pod, RTF dua tahap, Canary/Granite. |
| V Kesimpulan & Saran         | Belum mulai                   |                                                          |

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

## 5. Feedback dosen (checklist — dari `../proposal/feedback.txt`)

1. Jelaskan masalah *speaker dependency* di riset BCI. → **Sudah** di Batasan Bab I; bisa diperdalam di Bab II.
2. Tambah bagian di Bab II tentang deskripsi dataset + praproses. → **Bab II TODO.**
3. Jelaskan detail bagaimana sinyal diadaptasi ke input model. → **Bab II/III TODO.**

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
