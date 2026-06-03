# Draft Konten Slide — Bab III, IV, V (slide 13–24)

Sumber angka: `bab3.md`, `bab4.md`, `bab5.md`. Istilah Inggris ditulis *italic*.
Template visual mengikuti deck: judul kiri-atas, logo ITB kanan-atas, **bar biru `#344DB8`**
berisi label seksi (putih, tebal, rata tengah), kotak konten putih berbingkai, tabel dengan
baris kepala biru, gambar di kanan + kapsi di bawah, nomor halaman kanan-bawah.

Gambar yang dipakai (sudah ada di `docs/thesis/figures/`):
`fig_praproses.png`, `fig_dua_tahap.png`, `fig_e2e_llava.png`, `fig_e2e.png`, `lora.png`,
`fig_wer_vs_length.png`.

---

## Slide 13 — Praproses dan Ekstraksi Fitur

**Bar biru:** Praproses dan Ekstraksi Fitur
**Gambar:** `figures/fig_praproses_h.png` (versi dua kolom horizontal untuk slide) — *kapsi:* Alur praproses dan ekstraksi fitur

- Kedua arsitektur menerima input yang sama, yaitu matriks fitur neural berukuran **T × 256** per ujaran.
- Tiap *bin* 20 ms diwakili 256 nilai, yaitu 128 *threshold crossings* + 128 *spike band power* dari kanal area 6v.
- Fitur dinormalisasi dengan *z-score* per sesi untuk mengatasi variasi statistik antarsesi perekaman.
- Saat pelatihan ditambahkan *Gaussian smoothing* pada dimensi waktu serta augmentasi *white noise* dan *constant offset* agar model lebih tahan terhadap variasi sinyal.

*(Dataset dasar sudah dibahas di slide Bab II. Slide ini fokus ke langkah desain, yaitu normalisasi dan augmentasi.)*

---

## Slide 14 — Rancangan Arsitektur Dua Tahap

**Bar biru:** Rancangan Arsitektur Dua Tahap
**Gambar:** `figures/fig_dua_tahap_h.png` (versi dua baris horizontal untuk slide) — *kapsi:* Alur arsitektur dua tahap

- **Tahap 1 — Dekoder fonem.** Memetakan matriks fitur T × 256 menjadi probabilitas atas 40 kelas (39 fonem + 1 token jeda) ditambah token *blank* untuk CTC. Modul dapat diisi varian berbasis Transformer (Transformer murni, Conformer, atau Conformer + *spatial attention*).
- **Pelatihan dekoder.** *Loss* CTC, *optimizer* Adam, *batch* 32, *learning rate cosine* 0,04 → 0,004 dengan 1000 langkah *warmup*, hingga 150.000 langkah, *early stopping* berdasarkan PER validasi, presisi campuran.
- **Tahap 2 — Model bahasa (dua langkah):**
  - **(a)** Dekode *beam search* dengan model 5-*gram* berbasis WFST melalui *shallow fusion* → daftar *n-best*.
  - **(b)** *Rescoring* daftar *n-best* dengan model bahasa neural **LLaMA-2 7B**. Skor akhir = kombinasi berbobot skor akustik + skor LM neural + bonus penyisipan kata (bobot dicari lewat *grid search*).

---

## Slide 15 — Rancangan Arsitektur End-to-End

**Bar biru:** Rancangan Arsitektur End-to-End
**Gambar:** `figures/fig_e2e_llava_h.png` (gaya *LLaVA*, versi slide) dan `figures/fig_e2e.png` (*cross-attention*, Gambar III.5), tampilkan berdampingan.

- **Tiga komponen:** *encoder* Conformer + *spatial attention* (dimensi 512, *subsampling* ~4× pada waktu) → proyektor (*linear* + *layer normalization*, 512 → dimensi *hidden* FM) → dekoder FM. Keluaran proyektor disebut ***ECoG memory***.
- **Gaya *LLaVA* (FM teks, mis. Qwen).** Token *ECoG memory* dikonkatenasi di depan token teks, lalu diproses bersama. Risiko ***text shortcut***, yaitu model memprediksi teks hanya dari teks sebelumnya tanpa benar-benar memakai sinyal ECoG.
- **Cross-attention (FM audio, Whisper/Cohere).** Token teks melewati *self-attention* kausal, lalu mengakses *ECoG memory* via *cross-attention* (teks = *query*, ECoG = *key*/*value*). Sinyal ECoG tidak pernah masuk jendela *self-attention* teks sehingga *text shortcut* teratasi.
- **Tradeoff.** *LLaVA* lebih sederhana dan memakai LLM teks apa adanya, tetapi rawan *text shortcut*. *Cross-attention* lebih kokoh memaksa pemakaian sinyal ECoG, tetapi hanya berlaku untuk FM ber-*encoder-decoder*. Pada kedua jalur, dekoder FM diadaptasi dengan **LoRA**.

---

## Slide 16 — Pelatihan End-to-End

**Bar biru:** Pelatihan End-to-End
**Gambar (kanan):** `figures/lora.png` — *kapsi:* Skema LoRA (Hu et al., 2021)

- **Skema dua tahap:**
  - **(1) Pralatih *encoder*.** *Encoder* Conformer dilatih sendiri dengan CTC level-karakter (AdamW, LR 1×10⁻³ *cosine* + *warmup*, *weight decay* 0,01, *batch* 16).
  - **(2) Latih E2E utuh.** *Encoder* diinisialisasi dari hasil CTC, lalu *encoder* + proyektor ikut dilatih (LR kecil) sementara dekoder FM hanya di-LoRA. *Loss cross-entropy* hanya pada posisi token teks.
- **LoRA** diterapkan pada modul *attention* (q, k, v, o), *cross-attention*, dan *feed-forward* dekoder. *rank* 16, *alpha* 32, *dropout* 0,1.
- **Hiperparameter:** 15.000 langkah, *batch* efektif 16, 500 *warmup*, presisi campuran. LR puncak per kelompok: *encoder* 6,9×10⁻⁵, proyektor + *cross-attn* 1,0×10⁻³, LoRA 1,75×10⁻⁴.
- **Kenapa LoRA:** mengadaptasi FM besar secara efisien — hanya melatih sebagian kecil parameter dan **tanpa menambah latensi inferensi** karena bobot adapter dapat digabung ke bobot asal.

**Persamaan (taruh di dekat gambar):**  **W = W₀ + (α ⁄ r)·B·A**, dengan B = 0 dan A ∼ 𝒩(0, σ²) saat inisialisasi, serta *rank* r ≪ d.

---

## Slide 17 — Lingkungan Pengembangan

**Bar biru:** Lingkungan Pengembangan
**Tabel IV.1 (native):**

| Komponen | Spesifikasi |
| --- | --- |
| Prosesor | 16 vCPU (host AMD EPYC) |
| RAM | 62 GB |
| GPU | NVIDIA GeForce RTX 4090, 24 GB GDDR6X, Ada Lovelace, 16.384 *CUDA core* |
| Penyimpanan | 200 GB |
| Sistem Operasi | Ubuntu 22.04 LTS |
| Lingkungan virtual | Python *venv* (PyTorch CUDA 12 + TensorFlow 2.15) |

- Seluruh pelatihan dan evaluasi berjalan di GPU *cloud* **RunPod**. RTX 4090 dipilih karena ketersediaan, biaya sewa wajar, dan memori 24 GB cukup untuk FM besar dengan LoRA.
- Pembagian lingkungan: jalur E2E dan *rescoring* memakai PyTorch (transformers, peft); dekoder fonem dan dekode WFST memakai TensorFlow; metrik kesalahan dihitung dengan jiwer.

---

## Slide 18 — Evaluasi Dekoder Fonem

**Bar biru:** Evaluasi Dekoder Fonem
**Tabel IV.2 (native):**

| Dekoder Fonem | PER |
| --- | --- |
| GRU 1024 unit 5 lapisan | 0,1597 |
| Transformer murni | 0,2444 |
| Conformer vanila | 0,1477 |
| **Conformer + *spatial attention*** | **0,1428** |

- **Desain:** bandingkan empat dekoder fonem (GRU *baseline*, Transformer murni, Conformer vanila, Conformer + *spatial attention*) pada metrik **PER**, dengan *loss* CTC dan *batch* 32.
- **Pembahasan:** kedua varian Conformer mengalahkan GRU, sedangkan Transformer murni justru terburuk → modul konvolusi penting untuk menangkap pola lokal sinyal ECoG, *self-attention* murni belum cukup. *Spatial attention* menambah keunggulan dengan menangkap dependensi antarelektroda. Conformer + *spatial attention* (0,1428) juga mengalahkan dekoder GRU Seto (~0,192).

---

## Slide 19 — Evaluasi Arsitektur (Akurasi)

**Bar biru:** Evaluasi Arsitektur — Akurasi
**Tabel IV.3 penuh (native):**

| Arsitektur | WER | CER |
| --- | --- | --- |
| Dua tahap (GRU + 5-*gram*) | 0,1828 | 0,1327 |
| Dua tahap (Transformer + 5-*gram*) | 0,2927 | (menyusul) |
| Dua tahap (Conformer + 5-*gram*) | 0,1858 | 0,1253 |
| Dua tahap (GRU + 5-*gram* + LLaMA-2 7B) | 0,1638 | 0,1194 |
| **Dua tahap (Conformer + 5-*gram* + LLaMA-2 7B)** | **0,1556** | **0,1127** |
| E2E Qwen (gaya *LLaVA*) | 0,2537 | 0,2413 |
| E2E Whisper-medium.en | 0,1760 | 0,1508 |
| **E2E Whisper-large-v3** | **0,1716** | **0,1428** |
| E2E Cohere Transcribe | 0,1776 | 0,1523 |
| E2E Canary-Qwen | *tidak dievaluasi penuh* | *tidak dievaluasi penuh* |
| E2E Granite-Speech | *tidak dievaluasi penuh* | *tidak dievaluasi penuh* |

- **Desain:** bandingkan arsitektur utuh dua tahap vs E2E. Metrik **WER & CER** (*micro-average* tingkat korpus), data uji sebanding Willett. Canary-Qwen dan Granite-Speech diimplementasikan tetapi tidak dievaluasi penuh karena inkompatibilitas pustaka.
- **Pembahasan:**
  - Dua tahap + *rescoring* LLaMA-2 7B mencapai **WER terendah (0,1556)** dan mengalahkan Willett (~0,174) serta Seto (~0,169).
  - **Tanpa *rescoring* neural, semua dua tahap kalah dari Whisper-large-v3** (Conformer + 5g 0,1858, GRU + 5g 0,1828, Transformer + 5g 0,2927). Keunggulan dua tahap praktis berasal dari LLaMA, bukan dari pipeline fonemnya.
  - Antar-E2E: model audio *cross-attention* mengungguli FM teks *LLaVA* (Qwen tertinggal jauh, 0,2537), dan model audio lebih besar lebih baik (large-v3 > medium.en).

---

## Slide 20 — Penyimpanan dan Kecepatan

**Bar biru:** Penyimpanan dan Kecepatan
**Tabel IV.4 (native):**

| Arsitektur | Total Parameter (komponen terbesar) | Penyimpanan Total (komponen terbesar) | RTF | WPM |
| --- | --- | --- | --- | --- |
| Dua tahap (GRU + 5-*gram*) | 53,6 jt (GRU 53,55 jt) | 44,3 GB (TLG.fst 44,1 GB) | 0,0155 | 4.556 |
| Dua tahap (Transformer + 5-*gram*) | (menyusul) | (menyusul) | (menyusul) | (menyusul) |
| Dua tahap (Conformer + 5-*gram*) | 28,5 jt (Conformer 28,49 jt) | 44,2 GB (TLG.fst 44,1 GB) | 0,0069 | 10.307 |
| Dua tahap (GRU + 5-*gram* + LLaMA-2 7B) | 6,79 M (LLaMA-2 6,74 M) | 57,8 GB (TLG.fst 44,1 GB) | 0,0756 | 935 |
| Dua tahap (Conformer + 5-*gram* + LLaMA-2 7B) | 6,77 M (LLaMA-2 6,74 M) | 57,7 GB (TLG.fst 44,1 GB) | 0,0430 | 1.650 |
| E2E Qwen (gaya *LLaVA*) | ~0,86 M (FM 0,8 M) | 2,1 GB (FM 1,77 GB) | 0,0710 | 895 |
| E2E Whisper-medium.en | ~0,80 M (FM 769 jt) | 2,0 GB (FM 1,54 GB) | 0,0455 | 1.394 |
| E2E Whisper-large-v3 | ~1,57 M (FM 1,54 M) | 3,6 GB (FM 3,09 GB) | 0,0742 | 857 |
| E2E Cohere Transcribe | ~2,0 M (FM ~2 M) | 4,5 GB (FM 4,13 GB) | 0,0206 | 3.091 |
| E2E Canary-Qwen | ~4,2 M (encoder 2,5 M) | 9,9 GB (encoder 5,12 GB) | 0,1260 | 503 |
| E2E Granite-Speech | ~2,0 M (FM ~2 M) | 5,3 GB (FM 4,87 GB) | 0,0672 | 943 |

*(jt = juta, M = miliar)*

- Seluruh arsitektur memiliki **RTF ≪ 1** sehingga lebih cepat daripada durasi ucapan asli. Penyimpanan E2E **2,0–9,9 GB** vs dua tahap **44,2–57,8 GB**.
- **Pembahasan:**
  - Beban dua tahap hampir seluruhnya dari berkas **TLG.fst 44,1 GB** (ditambah LLaMA-2 7B pada varian *rescoring*), bukan dari dekoder fonem yang hanya ratusan MB. Inilah harga dari keunggulan akurasi 0,1556.
  - **Sambungan ke slide 19:** dua tahap Conformer + LLaMA unggul tipis di akurasi tetapi butuh 57,7 GB, sedangkan Whisper-large-v3 hanya 3,6 GB dengan akurasi kompetitif → E2E lebih seimbang untuk penerapan nyata.
  - Karena kecepatan semua sudah memadai, pertimbangan utama adalah penyimpanan dan akurasi. Cohere tercepat (WPM 3.091) walau parameter terbesar, karena dekodernya hanya 8 lapisan (vs 32 lapisan Whisper-large-v3).

---

## Slide 21 — Distribusi Probabilitas Fonem (IV.4.1)

**Bar biru:** Distribusi Probabilitas Fonem
**Tabel IV.5 (native):**

| Dekoder | Coverage | Oracle WER | Rata-rata *n-best* |
| --- | --- | --- | --- |
| Conformer + *spatial attention* | 58,0% | 0,1018 | 32,6 |
| GRU | 68,3% | 0,0796 | 48,3 |

- **Anomali:** PER Conformer lebih baik (0,1428 < 0,1597 GRU), tetapi pada dekode 5-*gram* tanpa *rescoring* WER GRU (0,1828) sedikit lebih rendah dari Conformer (0,1858).
- **Hipotesis dan uji:** distribusi fonem Conformer lebih tajam (entropi **0,0950 nats** vs GRU **0,1448 nats**, sekitar 34% lebih tajam) → *beam search* memangkas lebih agresif → jalur yang benar lebih sering hilang.
- **Pembahasan:** lebih dari 40% ujaran, transkrip benar tidak muncul di daftar *n-best* sehingga tidak terpulihkan oleh *rescorer* mana pun. Inilah *bottleneck* dua tahap. Keunggulan WER GRU hilang setelah *rescoring* LLaMA-2 7B, dan Conformer kembali unggul (0,1556 vs 0,1638).

---

## Slide 22 — Pengaruh Panjang Ujaran terhadap WER (IV.4.2)

**Bar biru:** Pengaruh Panjang Ujaran
**Gambar (kanan):** `figures/fig_wer_vs_length.png` — *kapsi:* WER per *bucket* panjang ujaran (Gambar IV.1)

- WER kedua arsitektur menurun seiring ujaran lebih panjang (ujaran 2-3 kata WER ~0,35, turun ke sekitar 0,14 pada 10 kata ke atas).
- **E2E Whisper-large-v3 unggul di hampir semua *bucket***, kecuali ujaran tepat **5 kata** (dua tahap 0,1750 vs E2E 0,2006).
- **Pembahasan:** model 5-*gram* berjendela tetap empat kata sebelumnya sehingga paling optimal tepat pada ujaran 5 kata. Pada ujaran lebih panjang, jendela tetap terbatas sehingga konteks tambahan menjadi keunggulan E2E. Dekoder *cross-attention* Whisper bersifat autoregresif tanpa jendela tetap dan memakai seluruh token yang sudah dibangkitkan.

---

## Slide 23 — Komplementaritas Kesalahan (IV.4.3)

**Bar biru:** Komplementaritas Kesalahan
**Tabel IV.6 (native):**

|  | Dua tahap benar | Dua tahap salah |
| --- | --- | --- |
| **E2E benar** | 175 (29,2%) | 71 (11,8%) |
| **E2E salah** | 66 (11,0%) | 288 (48,0%) |

- Perbandingan E2E Whisper-large-v3 vs dua tahap (Conformer + 5-*gram*) pada data uji.
- **22,8% ujaran** benar di satu arsitektur tetapi salah di yang lain → kedua arsitektur menangkap pola kesalahan berbeda.
- ***Best-of-two* WER = 0,1249**, jauh di bawah WER tunggal (E2E 0,1716; dua tahap 0,1858).
- **Pembahasan:** selisih ini menunjukkan potensi *ensembling* untuk mengangkat akurasi melebihi tiap arsitektur tunggal sebagai arah pengembangan selanjutnya.

---

## Slide 24 — Kesimpulan dan Saran (Bab V)

**Bar biru:** Kesimpulan dan Saran
**Tata letak:** dua kolom (Kesimpulan | Saran)

**Kesimpulan**
1. Arsitektur dua tahap berbasis Transformer berhasil dan mengalahkan Seto. Dekoder Conformer + *spatial attention* PER 0,1428, sistem utuh + *rescoring* LLaMA-2 7B WER 0,1556.
2. Arsitektur E2E berbasis FM berhasil. Whisper-large-v3 (*cross-attention*) WER 0,1716, mengalahkan semua dua tahap tanpa *rescoring* neural dengan penyimpanan jauh lebih kecil (3,6 GB vs 57,7 GB) → keseimbangan akurasi dan efisiensi lebih baik.
3. Analisis menyeluruh dilakukan dari sisi akurasi, efisiensi, pola kesalahan, dan keterbatasan. Dua tahap akurat tetapi berat dan ber-*ceiling* (oracle), E2E ringan dan kompetitif, serta keduanya saling melengkapi pada 22,8% ujaran.

**Saran**
- Tambah data latih dan pralatih *encoder* ECoG lintas-subjek.
- Kembangkan *rescorer* yang lebih baik (selisih *oracle* 0,1018 vs aktual 0,1556).
- Eksplorasi *ensembling* E2E + dua tahap (*best-of-two* 0,1249).
- Arah *speaker-independent*.
- Perluasan kosakata dan pengujian pada dataset lain.

*(Slide "Terima kasih" yang ada sekarang menjadi slide penutup setelah slide 24.)*
