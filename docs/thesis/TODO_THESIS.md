# TODO Tugas Akhir (revisi pasca-draf)

Status per 2026-06-09. Sumber: permintaan revisi setelah semua bab didraft.
Tanda: [x] selesai, [~] sebagian, [ ] belum.

---

## 1. Abstrak
- [ ] Tulis **ABSTRAK** (Bahasa Indonesia) ~200 kata + **Kata kunci**.
- [ ] Tulis **ABSTRACT** (Bahasa Inggris) sebagai terjemahannya (template ITB minta keduanya).
- Lokasi: berkas `abstrak.md` (belum dibuat) atau langsung di dokumen Word, sebelum Bab I.

## 2. Saran Bab V (V.2) — tambahan
- [ ] **Integrasi modalitas neural lain** (EEG, fMRI, ECoG, dan perekaman intrakortikal lain) menuju model fondasi neural lintas-modalitas. Argumen: jalur E2E berbasis FM mudah diperluas, cukup ganti/tambah *encoder* per modalitas.
- [ ] **Mengatasi bottleneck tahap pemetaan sinyal ke fonem dan *beam*** pada arsitektur dua tahap (akar dari *coverage* 58% / *oracle WER* 0,1018). Saran: tingkatkan dekoder fonem, kalibrasi ketajaman distribusi agar *beam* tidak over-prune, perlebar *beam* dan *n-best*.
- [ ] **Ubah saran *ensembling* menjadi rencana eksperimen langsung** (lihat §Catatan ensembling). Supervisor mengusulkan pemilihan berbasis skor keyakinan (*confidence*); terbuka untuk alternatif.

## 3. Penulisan istilah asing (miring / tidak)
- [x] Aturan diputuskan dan didokumentasikan di **`DAFTAR_ISTILAH.md`**.
- [x] Nama arsitektur/model di-non-miring-kan di bab1–bab4: `Transformer`, `Conformer`, `LLaVA`.
- [ ] Saat finalisasi ke Word, samakan seluruh dokumen dengan `DAFTAR_ISTILAH.md` (tidak diedit per-md sekarang, sesuai keputusan).

## 4. Formula / Persamaan
- [x] Penomoran **Persamaan II.1–II.5** ditambahkan di `bab2.md`.
- [x] Tiap persamaan sudah memiliki paragraf penjelasan (diverifikasi).
- [x] **Daftar Persamaan** dibuat di `DAFTAR_PERSAMAAN.md` (di Word dapat dibuat otomatis sebagai front matter).

## 5. Tambah konten (hindari kesan "happy path")
Kekhawatiran: penguji mengira pekerjaan sesederhana yang tertulis, padahal banyak proses tidak mulus.
- [ ] **Easy win — subbab penjelasan dedicated tiap Foundation Model** yang dipakai (Qwen3.5-0.8B, Whisper-medium.en, Whisper-large-v3, Cohere Transcribe, Canary-Qwen-2.5B, Granite-Speech-4.1-2B): arsitektur, modalitas latih, ukuran, alasan dipilih, cara adaptasi. Lokasi usul: `bab2.md` §II.5 atau `bab3.md` setelah Tabel III.1.
- [ ] Opsional — subbab/paragraf **tantangan dan keterbatasan implementasi**: Canary/Granite gagal dievaluasi penuh (inkompatibilitas pustaka), LM7 (LoRA-LLaMA) hilang dari disk, *overfitting* E2E pada data terbatas, *text shortcut* pada LLaVA. Membuat usaha riil terlihat.
- [ ] Opsional — detail praproses/augmentasi dan *hyperparameter tuning* yang dijalani (bukan hanya konfigurasi akhir).

## 6. Ensembling — pertimbangkan eksperimen nyata (lihat §Catatan)

## 7. Placeholder angka Bab IV — SELESAI (disinkronkan dari docx 2026-06-10)
- [x] Tabel IV.3: CER Dua tahap (Transformer + 5-gram) = **0,2094**.
- [x] Tabel IV.4: baris Dua tahap (Transformer + 5-gram) = **16,8 juta / 44,2 GB (TLG.fst 44,1 GB) / RTF 0,0044 / 6.573 WPM**.
- [x] Canary-Qwen & Granite-Speech **dievaluasi penuh** (keputusan user, docx = source of truth): Canary WER **0,2384** / CER **0,2115**; Granite **0,2337** / **0,2097**. Framing "tidak dievaluasi penuh" dihapus dari `bab4.md` & `SLIDES_BAB345.md`.
- [x] Tabel IV.4 Canary RTF dikoreksi dari 0,1260 → **0,3207** (sesuai docx).
- Catatan: **HANDOFF §8 kini usang** soal Canary/Granite "tidak dievaluasi penuh" — abaikan, pakai angka docx.

## 8. Konsistensi antarbab
- [ ] **Selaraskan framing *spatial attention* di Bab II §II.4.1.** Saat ini dibingkai "dapat dicoba karena berpotensi", padahal di Bab III.3.2 dan Bab IV sudah menjadi komponen yang benar-benar dipakai (mekanismenya kini dijelaskan). Samakan agar tidak kontradiktif.
- [ ] Sinkronkan daftar pustaka md ke dokumen Word saat finalisasi.

---

## Catatan ensembling (untuk diskusi/eksperimen)
Kedua model sudah ada. Target: turun di bawah WER tunggal terbaik (dua tahap+LLaMA 0,1556; E2E 0,1716) menuju batas atas oracle *best-of-two* ~0,1249. Gabungkan dua sistem terkuat (dua tahap+LLaMA dan E2E Whisper-large-v3); hitung ulang oracle pasangan ini dulu.

**Prasyarat lintas-metode (kalibrasi):** skor keyakinan kedua sistem TIDAK sebanding mentah. E2E = log-prob token (mis. ~-0,3/token); dua tahap = skor akustik + 5-gram (+LLaMA), skala sangat berbeda (mis. ~-64 per ujaran). Membandingkan langsung = apples-to-oranges sehingga "pilih confidence tertinggi" cenderung kolaps ke satu sistem. Perlu kalibrasi pada data validasi. Saat ini hanya keluaran data uji yang ter-cache, jadi set validasi perlu dijalankan ulang lewat kedua sistem.

### Peringkat metode (jika ada GPU + waktu)
1. **Seleksi berbasis confidence (router) — paling worth (★★★).** Per ujaran, pilih sistem yang diprediksi WER lebih rendah. Tangga pendekatan (laporkan sebagai ablasi):
   - (a) **argmax confidence mentah** — baseline tanpa latih; biasanya degenerate (kolaps ke satu sistem) karena skala beda.
   - (b) **kalibrasi per-model lalu argmax** — length-normalize + Platt/isotonic ke P(benar) atau perkiraan WER, baru pilih yang lebih tinggi.
   - (c) **regresi logistik / GBDT** atas fitur confidence (minimal 2 fitur = 1 per model; opsional + margin top1−top2, entropi, panjang, kesepakatan *n-best*). Belajar skala relatif + ambang otomatis (subsumsi (b)).
   - Ceiling = oracle 0,1249; realistis ~0,14–0,15. Cerita ablasi: argmax mentah gagal, kalibrasi/router berhasil.
2. **Fusi tingkat hipotesis (★★½, ceiling tertinggi).** Gabungkan *n-best* kedua sistem, normalisasi skor, lalu *rescore* union dengan LLaMA atau MBR. Bisa melampaui oracle seleksi karena memilih hipotesis yang tak ber-rank-1 di kedua sistem. Risiko: normalisasi skor heterogen.
3. **Fusi tingkat kata berbobot confidence (ROVER-style) (★½).** Menangkap komplementaritas intra-ujaran. Lemah untuk hanya 2 sistem (banyak seri). Lakukan terakhir.
- Lewati fusi tingkat logit (arsitektur CTC-fonem vs token Whisper terlalu heterogen).

### Kebutuhan GPU
- Logika seleksi/kalibrasi/router (argmax, Platt/isotonic, regresi logistik, k-fold): **CPU saja**, hitungan detik.
- **GPU hanya untuk sekali ekstraksi skor E2E (Whisper)**: jalankan generation dengan `return_dict_in_generate` + `output_scores=True` untuk log-prob per ujaran (dan *n-best* untuk metode 2), pada set validasi + uji. Skor dua tahap sudah ter-cache (CPU); decode validasi dua tahap = CPU/TF.
- Metode 2 (fusi hipotesis) juga butuh GPU untuk *rescoring* LLaMA.
- Tanpa GPU sama sekali: hanya bisa *one-sided gate* memakai confidence dua tahap yang sudah ter-cache (+ k-fold τ) atau *length router* (lihat catatan terpisah).

## Catatan ide konten tambahan (untuk diskusi)
- Ablasi kecil (mis. dampak *spatial attention*, dampak ukuran *beam*/*n-best*).
- Analisis kualitatif contoh keluaran (kasus benar/salah) tiap arsitektur.
- Pembahasan posisi terhadap SOTA Zhang/BIT 2025 (kenapa belum sebaik mereka: pralatih lintas-spesies + ensembling).

## Berkas terkait
- `DAFTAR_ISTILAH.md` — aturan + tabel istilah (miring/tidak).
- `DAFTAR_PERSAMAAN.md` — daftar persamaan.
- `SLIDES_BAB345.md` — draf konten slide Bab III–V.
- `HANDOFF.md` — konteks penulisan keseluruhan.

---

## Sudah selesai (sesi revisi 2026-06-10)
- **Abstrak** (Indonesia saja) dibuat di `abstrak.md` — latar belakang, tujuan, metode, hasil (PER 0,1428; dua tahap+LLaMA 0,1556; E2E 0,1716; *ensembling* 0,1441), saran.
- **Bab V.2 saran:** poin *ensembling* (kini sudah jadi hasil) **diganti** dengan dua saran baru — (3) keterbatasan tahap sinyal→fonem & *coverage* dan (4) integrasi modalitas neural lain (EEG/fMRI). Poin *speaker-independent* & kosakata digeser jadi 5 & 6.
- **Bab IV:** eksperimen *ensembling* ditulis sebagai **lanjutan IV.4.3 dalam bentuk paragraf (tanpa tabel)** — hanya *router* regresi logistik (WER **0,1441** vs 0,1556 dua tahap & 0,1716 E2E; *oracle* 0,1089). Tabel: *coverage* IV.5, kontingensi IV.6.
- **Keputusan huruf miring** untuk Transformer / Conformer / foundation model dicatat (lihat `DAFTAR_ISTILAH.md` & jawaban sesi).

### Perlu diputuskan saat alignment ke source of truth (`IF4092_...docx.pdf`)
- **Canary-Qwen & Granite-Speech:** docx (p63) sudah memuat angka — Canary WER 0,2384 / CER 0,2115; Granite 0,2337 / 0,2097 — sedangkan `bab4.md` Tabel IV.3 masih "(menyusul)" dan HANDOFF §8 menyebut keduanya "tidak dievaluasi penuh". **Konflik** — putuskan: pakai angka docx atau pertahankan "tidak dievaluasi penuh".
- **Jangan timpa revisi md** (mekanisme *spatial attention*, penomoran Persamaan, de-italic nama) saat align — docx belum memuatnya. Align = tarik DATA terbaru dari docx ke md, bukan sebaliknya.

## Sudah selesai (sesi revisi 2026-06-09)
- Penomoran **Persamaan II.1–II.5** di `bab2.md` + `DAFTAR_PERSAMAAN.md`.
- `DAFTAR_ISTILAH.md` (aturan huruf miring + tabel); nama Transformer/Conformer/LLaVA di-non-miring-kan di bab1–bab4.
- Mekanisme **spatial attention** (gating gaya SE, dim 64, 4 *head*) ditambahkan ke `bab3.md` §III.3.2; "dimensi 64" didisambiguasi di `bab4.md` §IV.2.1.
- Penjelasan "kenapa *rescoring* LLaMA memulihkan Conformer" ditambahkan ke slide 21 (`SLIDES_BAB345.md`).
- `SLIDES_BAB345.md`: draf konten slide 13–24 (Bab III–V), termasuk kolom Konfigurasi di tabel slide 18.
- Gambar slide horizontal: `figures/fig_praproses_h.png`, `figures/fig_dua_tahap_h.png`, `figures/fig_e2e_llava_h.png`; grafik slide 6 di `_slides_tmp/slide6_pipeline.png`.
