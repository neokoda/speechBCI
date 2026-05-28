# Thesis Writing Handoff

**Scope:** progress + context for *writing the thesis document* (Bahasa Indonesia). This is separate from the repo-root `HANDOFF.md`, which tracks *experiments*. When you need result numbers, the root `HANDOFF.md` and `docs/mds/EXPERIMENTS.md` are the source of truth.

**Last updated:** 2026-05-28 (Bab I drafted and under revision; rules files + bibliography established).

---

## 1. Files in this folder

| File                               | Role                                                                                                         |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `bab1.md`                        | BAB I PENDAHULUAN — drafted, under active revision by user + assistant.                                     |
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
| III Analisis & Rancangan     | Belum mulai                   | Dua tahap (Conformer) + varian E2E.                      |
| IV Implementasi & Eksperimen | Belum mulai                   | Sumber angka: root `HANDOFF.md`, `EXPERIMENTS.md`.   |
| V Kesimpulan & Saran         | Belum mulai                   |                                                          |

---

## 4. Keputusan framing Bab I (jaga konsistensi di bab lain)

- **Payung "berbasis Transformer".** Conformer disebut sebagai *varian* Transformer untuk domain ucapan, bukan paradigma terpisah. Menghindari "missing link" RNN→Conformer.
- **Speaker dependency** ada di **Batasan** (poin 1), bukan Latar Belakang. (Awalnya feedback dosen, tapi user memutuskan taruh di Batasan.)
- **Urutan argumen E2E** di §1.1: error propagation (pendukung) → dua keunggulan struktural (joint optimization + pengetahuan FM dipakai langsung dalam dekode, bukan rescoring terpisah) → kenapa FM (data terbatas) → skalabilitas/ekstensibilitas. **Jangan** menjadikan "error propagation hilang" sebagai klaim utama — secara empiris dua tahap masih menang, jadi itu akan dibantah hasil sendiri.
- **Latar Belakang = cerita masalah.** Detail implementasi (arsitektur konkret, slice, dsb.) ditunda ke Bab II/III. User eksplisit soal ini.
- **Related work (Feng 2024, Zhang/BIT 2025) ditunda ke Bab II §II.5.** Sengaja tidak di Latar Belakang.

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
