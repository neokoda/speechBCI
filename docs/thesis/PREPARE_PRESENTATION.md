# Persiapan Presentasi Tugas Akhir

Catatan belajar. Tiap bagian = satu pertanyaan yang belum dikuasai + jawaban singkat.

## Apakah *attention* benar-benar bisa menangkap hubungan spasial antarelektroda? Kalau berguna, kenapa?

**Berguna.** Intuisimu benar. Satu elektroda boleh "melihat" elektroda lain supaya informasinya masuk.

Kenapa perlu attend ke elektroda lain:

1. **Meredam derau.** Satu elektroda bisa berisik atau lemah di suatu *trial*. Beberapa elektroda yang merekam otot/artikulator yang sama bergerak bersama. Dengan menggabungkan mereka, sinyal yang benar diperkuat dan derau acak saling meniadakan.
2. **Menyorot elektroda penting.** Tidak semua elektroda informatif. Sebagian mati atau tidak relevan. *Attention* memberi bobot besar ke elektroda informatif dan bobot kecil ke yang tidak.

Contoh ilustrasi:

- Fonem /t/ dihasilkan lidah. Elektroda 5, 12, dan 40 kebetulan dekat area lidah dan aktif bersamaan saat /t/.
- Di *trial* ini elektroda 5 sedang berisik. Sendirian, ia bisa bikin model ragu.
- *Attention* membuat elektroda 5 "meminjam" info dari 12 dan 40 yang lebih bersih. Representasi /t/ jadi lebih kuat.
- Elektroda 80 ada di area tak relevan, jadi diberi bobot kecil dan diabaikan.

Catatan untuk *spatial attention* kita: modul kita bukan mencampur elektroda secara penuh, melainkan menghasilkan **gerbang (nilai 0 sampai 1) per elektroda**. Jadi bentuk konkretnya adalah "menimbang ulang" tiap kanal, menguatkan kanal informatif dan meredam kanal berisik, sambil tetap melihat hubungan antarelektroda saat menghitung bobotnya.

**Contoh di data kita (input, proses, output).** Ambil 4 elektroda selama 3 *bin* (aslinya 256 elektroda, T *bin*). Input = matriks fitur.

```
        e1    e2    e3    e4
bin1    0,2   1,8   0,1   2,0
bin2    0,3   1,5   0,0   1,9
bin3    0,1   1,7   0,2   2,1
```

Langkah:

1. **Ringkas tiap elektroda** = rata-rata terhadap waktu (1 angka per elektroda):

```
e1 = (0,2+0,3+0,1)/3 = 0,2
e2 = (1,8+1,5+1,7)/3 = 1,667
e3 = (0,1+0,0+0,2)/3 = 0,1
e4 = (2,0+1,9+2,1)/3 = 2,0
ringkasan = [0,2 , 1,667 , 0,1 , 2,0]
```

2. **Jadikan tiap elektroda sebuah "token".** Angka ringkasan tiap elektroda diproyeksikan jadi vektor (asli dim 64, di sini dim 2 biar mudah) lalu ditambah *embedding* identitas elektroda (dipelajari, beda tiap elektroda, biar model tahu ini elektroda nomor berapa).

```
              proyeksi [s,s]    + identitas     = token
e1 (0,2)      [0,2 , 0,2]       [0,0 , 0,5]       [0,2 , 0,7]
e2 (1,667)    [1,667 , 1,667]   [0,5 , 0,0]       [2,167 , 1,667]
e3 (0,1)      [0,1 , 0,1]       [0,1 , 0,1]       [0,2 , 0,2]
e4 (2,0)      [2,0 , 2,0]       [0,2 , 0,3]       [2,2 , 2,3]
```

3. **Self-attention antarelektroda.** Ini **mekanisme yang sama persis** dengan bagian Transformer di atas (Q, K, V, softmax). Bedanya, yang jadi "token" adalah elektroda, bukan langkah waktu. Itu sebabnya disebut attention *spasial*. Contoh untuk e1 (pakai W_Q=W_K=W_V=identitas biar fokus ke idenya, jadi Q=K=V=token):

Skor = t1·tj / √2 (√2 = 1,414):

```
t1·t1 = 0,53   -> 0,375
t1·t2 = 1,600  -> 1,132
t1·t3 = 0,18   -> 0,127
t1·t4 = 2,05   -> 1,450
```

softmax -> bobot perhatian e1 = `[0,146 , 0,312 , 0,114 , 0,428]`. Jadi e1 paling melihat e4 (0,428) dan e2 (0,312), yaitu elektroda yang aktif.

Keluaran e1 = jumlah berbobot token:

```
out_e1 = 0,146*t1 + 0,312*t2 + 0,114*t3 + 0,428*t4 = [1,670 , 1,629]
```

(4 *head* = langkah ini diulang 4 kali dengan bobot beda lalu hasilnya digabung. Ulangi juga untuk e2, e3, e4.)

4. **Ubah keluaran jadi gerbang.** Vektor keluaran tiap elektroda dilewatkan lapisan linear kecil (vektor -> 1 angka) lalu *sigmoid* (memaksa nilai jadi 0 sampai 1):

```
raw = out_e1 · [0,5 , 0,5] + bias(-1,5) = 0,835 + 0,815 - 1,5 = 0,150
gate_e1 = sigmoid(0,150) = 1 / (1 + e^-0,150) = 0,537
```

Semua bobot (proyeksi, identitas, Q/K/V, dan gerbang) **dipelajari saat latih**. Setelah latih, elektroda sepi/berisik cenderung dapat gerbang rendah dan elektroda informatif dapat gerbang tinggi. Angka `gate = [0,30 ; 0,95 ; 0,20 ; 0,98]` di langkah 5 hanya contoh hasil akhir. Bagian ini menunjukkan dari mana angka gerbang itu berasal.

5. **Timbang ulang**: kalikan tiap kolom input asli dengan gerbangnya (kolom e1 ×0,30 ; e2 ×0,95 ; e3 ×0,20 ; e4 ×0,98):

```
        e1     e2      e3     e4
bin1    0,06   1,710   0,02   1,960
bin2    0,09   1,425   0,00   1,862
bin3    0,03   1,615   0,04   2,058
```

Output ini (256 kanal yang sudah ditimbang ulang) baru masuk ke blok Conformer. Jadi elektroda berisik/sepi diredam dulu, elektroda informatif diperkuat.

## Ilustrasi Transformer + perhitungan angka nyata (QKV, softmax, dst.)

Contoh kecil: 2 token, dimensi model = 2, 1 *head*. (Angka ilustratif.)

**Rumus inti:**

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

**Langkah 0 — input** (anggap sudah *embedding* + *positional*):

```
x1 = [1, 0]
x2 = [0, 1]
```

**Dari mana dimensi tiap bobot?**

```
d_model = 2  -> panjang tiap vektor input (x1, x2). Di contoh ini kita pilih 2.
d_k = 2      -> panjang vektor Q dan K. Q dan K WAJIB sama panjang, sebab dihitung dot product q·k.
d_v = 2      -> panjang vektor V (boleh beda dari d_k; di sini kita pakai 2).
```

Aturan bentuk matriks: `input (1 × d_model) · W (d_model × d_keluaran) = hasil (1 × d_keluaran)`. Jadi tiap matriks bobot memetakan dari `d_model` ke dimensi keluaran yang diinginkan:

```
W_Q : (d_model × d_k) = (2 × 2)
W_K : (d_model × d_k) = (2 × 2)
W_V : (d_model × d_v) = (2 × 2)
```

Ketiganya 2×2 karena d_model, d_k, dan d_v semuanya 2. Kalau *multi-head*, biasanya d_k = d_model / jumlah head. Di sini 1 *head*, jadi d_k = d_model = 2.

**Langkah 1 — proyeksi Q, K, V** (kalikan input dengan matriks bobot):

```
W_Q = [[2,0],[0,1]]   W_K = [[1,0],[0,1]]   W_V = [[1,2],[3,0]]

q1 = x1·W_Q = [2, 0]      k1 = [1, 0]      v1 = [1, 2]
q2 = x2·W_Q = [0, 1]      k2 = [0, 1]      v2 = [3, 0]
```

**Langkah 2 — skor = Q·Kᵀ, lalu dibagi √d_k** (d_k = 2, √2 = 1,414):

```
Token 1:
  q1·k1 = 2*1 + 0*0 = 2      -> 2 / 1,414 = 1,414
  q1·k2 = 2*0 + 0*1 = 0      -> 0 / 1,414 = 0

Token 2:
  q2·k1 = 0*1 + 1*0 = 0      -> 0
  q2·k2 = 0*0 + 1*1 = 1      -> 1 / 1,414 = 0,707
```

**Langkah 3 — softmax tiap baris** (softmax([a,b]) = [eᵃ, eᵇ] / (eᵃ+eᵇ)):

```
Token 1: softmax([1,414, 0])
  e^1,414 = 4,113 ; e^0 = 1 ; jumlah = 5,113
  bobot = [0,804 , 0,196]

Token 2: softmax([0, 0,707])
  e^0 = 1 ; e^0,707 = 2,028 ; jumlah = 3,028
  bobot = [0,330 , 0,670]
```

**Langkah 4 — keluaran = bobot · V:**

```
out1 = 0,804*v1 + 0,196*v2 = 0,804*[1,2] + 0,196*[3,0] = [1,392 , 1,608]
out2 = 0,330*v1 + 0,670*v2 = 0,330*[1,2] + 0,670*[3,0] = [2,340 , 0,660]
```

Artinya token 1 lebih fokus ke dirinya sendiri (bobot 0,804), token 2 lebih fokus ke token 2 (bobot 0,670).

**Langkah 5 — sisa blok** (rumus, tanpa angka biar ringkas):

```
1. Add & Norm : z = LayerNorm(x + out)      (residual + normalisasi)
2. Feed-Forward: f = ReLU(z·W1 + b1)·W2 + b2  (perdalam representasi)
3. Add & Norm : keluaran = LayerNorm(z + f)
```

**Multi-head**: langkah 1-4 diulang beberapa kali dengan W_Q/W_K/W_V berbeda, hasil tiap *head* digabung (*concat*) lalu diproyeksikan lagi. Tujuannya tiap *head* menangkap pola hubungan berbeda.

## Apa itu *threshold crossings* dan *spike band power*? + contoh konversi dari sinyal mentah (termasuk *binning*)

Sinyal mentah = tegangan listrik dari elektroda, direkam sangat cepat (misal 30.000 sampel per detik / 30 kHz).

**Binning**: sinyal dipotong jadi jendela pendek selebar 20 ms. Pada 30 kHz, satu *bin* = 0,020 × 30.000 = **600 sampel**. Tiap fitur dihitung per *bin*, per kanal.

**1. Threshold crossings (TX)** = berapa kali tegangan menembus suatu ambang batas dalam satu *bin*. Ambangnya −3,5 × RMS sinyal (di sini anggap −100 µV). "Menembus" = turun dari atas ambang ke bawah ambang.

**2. Spike band power (SBP)** = rata-rata tegangan kuadrat dalam satu *bin*, setelah sinyal disaring *high-pass* di atas 250 Hz. Ukuran seberapa "besar" aktivitas *spike*.

Contoh (pakai 10 sampel biar mudah, ambang = −100 µV):

```
sampel (µV): -20  -50  -120  -80  -110  -30   10  -105  -60  -90
posisi thd :  atas atas BAWAH atas BAWAH atas atas BAWAH atas atas
```

**Hitung TX** (setiap kali "atas -> BAWAH"):

```
-50  -> -120  : crossing 1
-80  -> -110  : crossing 2
10   -> -105  : crossing 3
TX = 3
```

**Hitung SBP** (rata-rata kuadrat sampel):

```
kuadrat: 400, 2500, 14400, 6400, 12100, 900, 100, 11025, 3600, 8100
jumlah = 59.525
SBP = 59.525 / 10 = 5.952,5 µV²
```

Jadi satu *bin* untuk satu kanal menghasilkan **2 angka**: TX = 3 dan SBP = 5.952,5.

Untuk 128 kanal, satu *bin* = 128 nilai TX + 128 nilai SBP = **vektor 256 dimensi**. Tumpuk semua *bin* dalam satu ujaran → **matriks fitur T × 256** (T = jumlah *bin*).

## Lanjutan: dari matriks fitur ke input model (z-score, Gaussian smoothing, white noise, constant offset)

Pakai satu kanal (nilai TX) selama 5 *bin* dalam satu sesi. *Bin* pertama = 3 (dari contoh di atas).

```
x = [3, 8, 2, 9, 3]
```

**Langkah A — Normalisasi z-score (per sesi, per kanal).** Rumus: `z = (x − mean) / std`. Hitung untuk kanal ini di seluruh sesi.

```
mean = (3+8+2+9+3)/5 = 5
selisih   : [-2, 3, -3, 4, -2]
kuadrat   : [4, 9, 9, 16, 4]  -> jumlah 42
varians   = 42/5 = 8,4  -> std = 2,898

z = [(3-5)/2,898 , (8-5)/2,898 , (2-5)/2,898 , (9-5)/2,898 , (3-5)/2,898]
z = [-0,690 , 1,035 , -1,035 , 1,380 , -0,690]
```

Tujuan: tiap kanal jadi mean 0 dan ragam 1, supaya beda skala antarsesi hilang.

**Langkah B — Gaussian smoothing** (saat latih). Konvolusi tiap kanal pada dimensi waktu dengan *kernel* Gauss, misal `[0,25 , 0,50 , 0,25]`. Nilai baru = rata-rata berbobot tetangganya. Rumus: `s[i] = 0,25·z[i-1] + 0,50·z[i] + 0,25·z[i+1]`.

```
s[1] = 0,25*(-0,690) + 0,50*(1,035)  + 0,25*(-1,035) = 0,086
s[2] = 0,25*(1,035)  + 0,50*(-1,035) + 0,25*(1,380)  = 0,086
s[3] = 0,25*(-1,035) + 0,50*(1,380)  + 0,25*(-0,690) = 0,259
(ujung s[0], s[4] pakai padding tepi)

s ≈ [-0,259 , 0,086 , 0,086 , 0,259 , -0,173]
```

Lihat: lonjakan tajam (1,380) jadi halus (0,259). Tujuannya meredam fluktuasi cepat yang bukan sinyal niat bicara.

**Langkah C — White noise** (augmentasi, hanya saat latih). Tambah derau acak dari distribusi normal (misal σ = 0,1) ke tiap nilai. Rumus: `x_baru = x + noise`.

```
noise (contoh) : [ 0,05 , -0,08 , 0,12 , -0,03 , 0,09 ]
hasil          : [-0,209 , 0,006 , 0,206 , 0,229 , -0,083]
```

Tujuan: model tidak terlalu bergantung pada nilai persis, jadi lebih tahan variasi.

**Langkah D — Constant offset** (augmentasi, hanya saat latih). Tambah **satu** nilai konstan yang sama ke SELURUH *bin* kanal itu (misal +0,20). Meniru pergeseran garis dasar sinyal.

```
+0,20 ke semua : [-0,009 , 0,206 , 0,406 , 0,429 , 0,117]
```

**Hasil akhir**: deretan angka ini menjadi satu kolom (satu kanal) dari matriks **T × 256** yang masuk ke model. Semua 256 kanal diproses sama. Saat inferensi, langkah C dan D dilewati (augmentasi hanya untuk latih). Pemendekan sekuens (*subsampling*) tidak dilakukan di sini karena itu bagian dari arsitektur model, bukan praproses.

Ringkas urutannya:

```
matriks fitur T×256
  -> z-score (selalu)
  -> Gaussian smoothing (latih)
  -> + white noise (latih)
  -> + constant offset (latih)
  -> masuk ke model
```

## Apa itu WFST? Bentuknya seperti apa? + proses *rescoring* n-best pakai LLaMA (contoh angka + rumus)

**WFST (weighted finite-state transducer)** = graf berbobot. Terdiri atas **state** (lingkaran) dan **arc** (panah). Tiap arc berlabel `input : output / bobot`. Kita menelusuri graf dari awal ke akhir, menerjemahkan input jadi output sambil menjumlahkan bobot. Bobot biasanya `−log(probabilitas)`, jadi makin kecil makin bagus.

Contoh WFST tata bahasa (G) mini untuk kalimat "i am ...":

```
        i/0,7          am/0,3          better/1,2
  (0) ---------> (1) ---------> (2) ---------------> (3)
                                  \                    ^
                                   \    bettor/1,1     |
                                    \-----------------/

Jalur "i am better" : 0,7 + 0,3 + 1,2 = 2,2
Jalur "i am bettor" : 0,7 + 0,3 + 1,1 = 2,1  <- lebih kecil, jadi 5-gram salah pilih "bettor"
```

Di sistem lengkap, tiga graf digabung (*compose*) jadi satu TLG:

```
T : keluaran CTC   -> fonem
L : fonem          -> kata
G : n-gram         -> bobot antarkata
```

Lalu *beam search* mencari jalur berbobot terbaik → menghasilkan **daftar n-best** (beberapa kalimat kandidat).

**Rescoring dengan LLaMA.** Daftar n-best dinilai ulang. Skor akhir tiap hipotesis:

```
skor_akhir = s_ak · log p_akustik + α · log p_neural + β · N_kata
```

- `log p_akustik` = skor dari dekoder fonem.
- `log p_neural`  = skor dari LLaMA (lihat seluruh konteks kalimat).
- `N_kata`        = jumlah kata (bonus biar kalimat tak terlalu pendek).
- `s_ak, α, β`    = bobot (misal 0,5 ; 0,8 ; 0,5).

Contoh angka (ilustratif) untuk ujaran "i am better":

```
Hipotesis        log p_akustik   log p_neural(LLaMA)   N_kata
i am bettor          -60,0            -18,0               3
i am better          -60,5             -9,0               3
i'm better           -61,0            -10,0               2
i am butter          -62,0            -20,0               3
```

Hitung skor akhir (s_ak=0,5 ; α=0,8 ; β=0,5):

```
i am bettor : 0,5*(-60,0) + 0,8*(-18,0) + 0,5*3 = -30,0 -14,4 +1,5 = -42,9
i am better : 0,5*(-60,5) + 0,8*(-9,0)  + 0,5*3 = -30,25 -7,2 +1,5 = -35,95  <- tertinggi
i'm better  : 0,5*(-61,0) + 0,8*(-10,0) + 0,5*2 = -30,5  -8,0 +1,0 = -37,5
i am butter : 0,5*(-62,0) + 0,8*(-20,0) + 0,5*3 = -31,0 -16,0 +1,5 = -45,5
```

Sebelum *rescoring*, 5-gram menaruh "i am bettor" di peringkat 1. Setelah *rescoring*, LLaMA tahu "i am better" jauh lebih lazim (log p_neural −9,0 vs −18,0), jadi skornya naik jadi tertinggi (−35,95) dan **"i am better" terpilih**.

## Alur dan perhitungan LLaVA (E2E untuk FM teks), dari ECoG ke teks

Dipakai untuk FM teks *decoder-only* (Qwen). Idenya: sinyal ECoG diubah jadi "token" lalu **ditaruh di depan token teks**, semua diproses bareng oleh dekoder.

**Langkah 1 — encoder + proyektor jadi "ECoG memory".** Sinyal ECoG T×256 masuk *encoder* Conformer, hasilnya diproyeksikan (lapisan *linear*) ke dimensi FM. (Angka ilustratif, dim 2.)

```
keluaran encoder : h1 = [0,5 , 1,0]   h2 = [1,5 , 0,2]
W_proj = [[1 , 0,5] , [0,5 , 1]]

e1 = h1·W_proj = [0,5*1 + 1,0*0,5 , 0,5*0,5 + 1,0*1] = [1,0 , 1,25]
e2 = h2·W_proj = [1,5*1 + 0,2*0,5 , 1,5*0,5 + 0,2*1] = [1,6 , 0,95]
```

`e1, e2` = ECoG memory (token sinyal).

**Langkah 2 — konkatenasi gaya LLaVA.** Token ECoG ditaruh di DEPAN token teks. Misal token awal teks `<bos>` = `t0 = [0,2 , 0,3]`.

```
urutan masukan dekoder : [ e1 , e2 , t0 ]
```

**Langkah 3 — self-attention kausal.** Tiap posisi melihat dirinya dan semua yang di depannya. Waktu memprediksi kata pertama, `t0` melihat `e1, e2, t0`. (Pakai Q=K=V=identitas biar fokus.)

```
skor t0 (dibagi √2):
  t0·e1 = 0,575 -> 0,407
  t0·e2 = 0,605 -> 0,428
  t0·t0 = 0,130 -> 0,092
softmax -> bobot = [0,363 , 0,371 , 0,265]

out_t0 = 0,363*e1 + 0,371*e2 + 0,265*t0 = [1,010 , 0,886]
```

Jadi info dari ECoG (e1, e2) masuk ke `t0` lewat self-attention biasa.

**Langkah 4 — prediksi kata.** `out_t0` (setelah FFN) dikalikan matriks kosakata jadi *logit* tiap kata, lalu softmax.

```
vektor kata: "i"=[1,0 , 0,8]  "am"=[0,2 , 0,1]  "the"=[0,9 , 0,2]
logit "i"   = 1,010*1,0 + 0,886*0,8 = 1,719   <- tertinggi
logit "am"  = 1,010*0,2 + 0,886*0,1 = 0,291
logit "the" = 1,010*0,9 + 0,886*0,2 = 1,086
```

Kata pertama = "i". **Langkah 5 — autoregresif**: "i" ditambah ke urutan, ulangi langkah 3-4 untuk kata berikut ("am"), dan seterusnya sampai token akhir.

**Catatan (text shortcut).** Karena token teks dan token ECoG ada di jendela self-attention yang sama, token teks bisa langsung melihat token teks sebelumnya. Risikonya model menebak teks dari teks saja tanpa benar-benar memakai ECoG. Ini yang diperbaiki oleh *cross-attention* di bawah.

## Alur dan perhitungan cross-attention (E2E untuk FM audio), dari ECoG ke teks

Dipakai untuk FM audio *encoder-decoder* (Whisper, Cohere). Idenya: ECoG memory TIDAK dicampur ke token teks, tetapi diakses lewat jalur terpisah.

**Langkah 1-2 sama seperti LLaVA**: encoder + proyektor menghasilkan ECoG memory `e1 = [1,0 , 1,25]`, `e2 = [1,6 , 0,95]`. Bedanya, ECoG memory ini menggantikan keluaran *encoder* audio bawaan Whisper.

**Langkah 3 — di tiap lapisan dekoder, dua tahap:**

*(a) Self-attention kausal HANYA antar token teks.* ECoG tidak ikut di sini. Misal representasi `<bos>` setelah tahap ini = `s = [0,4 , 0,5]`.

*(b) Cross-attention.* Token teks jadi **Query**, ECoG memory jadi **Key dan Value**.

```
q = s = [0,4 , 0,5]   ;   k1=v1=e1   ;   k2=v2=e2

skor (dibagi √2):
  q·e1 = 1,025 -> 0,725
  q·e2 = 1,115 -> 0,789
softmax -> bobot = [0,484 , 0,516]

cross_out = 0,484*e1 + 0,516*e2 = [1,310 , 1,095]
```

**Langkah 4 — prediksi kata** (sama seperti LLaVA, `cross_out` -> logit -> softmax):

```
logit "i"   = 1,310*1,0 + 1,095*0,8 = 2,186   <- tertinggi
logit "am"  = 1,310*0,2 + 1,095*0,1 = 0,372
logit "the" = 1,310*0,9 + 1,095*0,2 = 1,398
```

Kata pertama = "i", lalu autoregresif seperti biasa.

**Beda utama dengan LLaVA:** di sini ECoG tidak pernah masuk jendela self-attention teks. Satu-satunya jalan dari ECoG ke prediksi teks adalah *cross-attention*. Jadi model dipaksa memakai ECoG, tidak bisa curang (*text shortcut* hilang).

## Perhitungan dan pelatihan LoRA (rank, alpha, dropout, target q/k/v/o)

**Ide.** Bobot asli FM `W0` dibekukan. Perubahan untuk tugas baru ditambahkan lewat dua matriks kecil `A` dan `B`. Rumus:

```
W = W0 + ΔW = W0 + (α / r) · B · A
```

- `W0` : bobot asli, dibekukan, ukuran d×k.
- `A`  : ukuran r×k, `B` : ukuran d×r. `r` (rank) kecil.
- `α`  : faktor skala. Skala efektif = α/r.

**Contoh perhitungan** (d=2, k=2, r=1, α=2, jadi α/r=2):

```
W0 = [[2,0],[0,2]]   (beku)
A  = [[0,1 , 0,3]]      (1×2)
B  = [[0,5],[0,2]]      (2×1)
x  = [1 , 1]            (masukan lapisan)
```

Jalur LoRA dihitung dari kanan ke kiri:

```
A·x        = [0,1*1 + 0,3*1] = [0,4]                (turun ke dim r=1)
B·(A·x)    = [0,5 , 0,2] * 0,4 = [0,2 , 0,08]        (naik ke dim d=2)
(α/r)·...  = 2 * [0,2 , 0,08] = [0,4 , 0,16]         (skala)

jalur asli : W0·x = [2 , 2]
keluaran h = W0·x + jalur LoRA = [2,4 , 2,16]
```

Jadi LoRA cuma menambah koreksi kecil `[0,4 , 0,16]` ke keluaran asli `[2 , 2]`.

**Rumus dipakai pas apa.** Jalur `(α/r)BA·x` ditambahkan setiap *forward pass*, baik saat latih maupun inferensi. Setelah latih, `(α/r)BA` bisa digabung ke `W0` jadi satu matriks, sehingga tidak menambah latensi.

**Pelatihan LoRA.** Hanya `A` dan `B` yang dilatih, `W0` beku. Di awal `B = 0` sehingga `ΔW = 0` (model mulai sama persis dengan aslinya), lalu `A` dan `B` berangsur belajar. Jumlah parameter dilatih = `r·(d+k)`, jauh lebih kecil dari `d·k`. Contoh nyata: `W` 4096×4096 = 16,7 juta; LoRA r=16 = 16·(4096+4096) = 131 ribu (~0,8%).

**Kapan rank, alpha, dropout dipakai:**

- **rank r** = besar "leher botol" `A`/`B`. Makin besar r, makin banyak kapasitas dan parameter. Dipakai saat memfaktorkan `ΔW = BA`.
- **alpha α** = kekuatan pengaruh LoRA. Dipakai sebagai pengali `α/r` sebelum `BA` ditambahkan. r=16, α=32 -> skala 2.
- **dropout** = dipakai HANYA saat latih, pada masukan jalur LoRA (`x` sebelum `A`). Sebagian nilai `x` diacak jadi nol untuk mencegah *overfitting*. Jadi jalurnya `B·A·dropout(x)·(α/r)`. Saat inferensi dropout mati.

**Target q/k/v/o.** Pada attention ada 4 matriks bobot, yaitu `W_Q`, `W_K`, `W_V`, dan `W_O` (proyeksi keluaran). "Target q,k,v,o" berarti tiap matriks ini dikasih pasangan LoRA sendiri:

```
Q = x·(W_Q0 + (α/r)·B_q·A_q)
K = x·(W_K0 + (α/r)·B_k·A_k)
V = x·(W_V0 + (α/r)·B_v·A_v)
O = x·(W_O0 + (α/r)·B_o·A_o)
```

Semua `W_*0` beku, cuma 4 pasang `A/B` kecil yang dilatih (plus *cross-attention* dan FFN pada rancangan kita). Jadi model belajar menyesuaikan Q, K, V, dan O untuk sinyal ECoG tanpa mengutak-atik bobot asli FM.

## Gunanya alpha, rank, A, dan B di LoRA apa?

Kita mau mengubah bobot `W0` yang besar, tapi tanpa melatih ulang semuanya. Solusinya, perubahan `ΔW` ditiru dengan dua matriks kecil dan satu tombol skala.

- **A (matriks kompres).** Menurunkan masukan dari dimensi penuh `k` ke dimensi kecil `r`. Ibarat memampatkan info ke ringkasan kecil. Contoh: `A·x` mengubah `[1,1]` jadi `[0,4]` (dim 2 -> dim 1).
- **B (matriks ekspansi).** Menaikkan kembali dari `r` ke dimensi penuh `d`. Contoh: `B·[0,4]` jadi `[0,2 , 0,08]` (dim 1 -> dim 2). Gabungan `B·A` inilah pengganti murah untuk `ΔW`. Hanya A dan B yang dilatih. B mulai dari nol supaya latihan dimulai dari model asli tanpa perubahan.
- **rank r.** Ukuran "leher botol" antara A dan B. Ini yang bikin LoRA murah, sebab `ΔW` dipaksa berpangkat rendah. r kecil = sedikit parameter dan kapasitas kecil. r besar = lebih banyak parameter dan kapasitas besar.
- **alpha α.** Tombol seberapa kuat perubahan LoRA berpengaruh. Skala efektifnya `α/r`, dikalikan ke `B·A` sebelum ditambahkan. α besar = pengaruh LoRA lebih kuat. Gunanya, kita bisa atur kekuatan tanpa melatih ulang A dan B.

Singkatnya: **A dan B** yang belajar perubahannya (murah karena kecil), **rank** menentukan seberapa kecil, **alpha** menentukan seberapa kuat perubahan itu dipakai.

## d = 512 di Transformer dan Conformer itu apa?

`d` (sering ditulis `d_model` atau "dimensi *hidden*") = **panjang vektor yang mewakili tiap langkah waktu di dalam model**. Ini "lebar" model. Semua representasi internal berbentuk vektor sepanjang 512.

Alurnya di data kita:

```
tiap bin masuk : vektor 256 (128 TX + 128 SBP)
  -> dipetakan (lapisan linear) jadi vektor 512   <- mulai dari sini d = 512
  -> semua blok Transformer/Conformer bekerja di ruang 512
  -> keluaran tiap bin tetap vektor 512
```

Jadi kalau satu ujaran punya T bin, di dalam model bentuknya matriks `T × 512` (tiap baris = 1 bin, panjang 512).

Kaitannya dengan angka lain:

- **Attention**: Q, K, V juga berdimensi 512, dibagi ke *head*. Kalau 8 *head*, tiap *head* pegang 512 / 8 = 64 (`d_k = 64`).
- **Feed-forward**: biasanya melebar dulu ke 2048 lalu balik ke 512.
- Makin besar `d`, kapasitas model makin besar tapi parameter dan komputasi juga makin banyak.

**Jangan tertukar**: `d = 512` ini lebar *encoder* utama. Angka `64` pada modul *spatial attention* di atas beda hal, itu lebar representasi internal modul kecil itu saja, bukan `d_model`.

## `β · N_kata` (bonus penyisipan kata) itu buat apa?

**Masalahnya**: model bahasa cenderung memberi skor lebih tinggi ke kalimat yang lebih pendek. Sebabnya, tiap kata menambah satu peluang yang nilainya kurang dari 1, jadi makin banyak kata makin kecil peluang totalnya (skor makin negatif). Akibatnya sistem jadi condong **membuang kata**.

**Solusinya**: tambahkan bonus `β` untuk setiap kata. Jadi hipotesis yang lebih panjang tidak dirugikan hanya karena punya lebih banyak kata. `β` = besar bonus, `N_kata` = jumlah kata.

Contoh (α=0,8 untuk skor LM, β=0,5). Bandingkan yang benar vs yang menghilangkan kata "i":

```
                 log p_neural   N_kata
i am better         -9,0          3
am better           -8,5          2   (lebih pendek, skor LM lebih tinggi)
```

Tanpa bonus (cuma `α · log p_neural`):

```
i am better : 0,8*(-9,0) = -7,2
am better   : 0,8*(-8,5) = -6,8   <- menang, padahal salah (kata "i" hilang)
```

Dengan bonus (`α · log p_neural + β · N_kata`):

```
i am better : -7,2 + 0,5*3 = -5,7   <- sekarang menang (benar)
am better   : -6,8 + 0,5*2 = -5,8
```

Jadi bonus ini mengimbangi kecenderungan model bahasa memilih kalimat pendek, supaya kata yang seharusnya ada tidak ikut terbuang.

## Bobot akustik, bobot neural, dan bonus kata (s_ak, α, β) ditentukan bagaimana?

Bukan ditebak asal, tapi **dicari lewat *grid search* pada data validasi**. Bobot-bobot ini *hyperparameter* (disetel), bukan bobot model yang dilatih dengan *backpropagation*.

Prosesnya:

1. Tentukan beberapa nilai calon untuk tiap bobot. Contoh:

```
s_ak (skala akustik) : {0,3 ; 0,5 ; 0,8}
α    (bobot neural)  : {0,6 ; 0,8 ; 1,0}
β    (bonus kata)    : {0 ; 0,5 ; 1,0}
```

2. Coba SEMUA kombinasi (di contoh ini 3×3×3 = 27 kombinasi).
3. Untuk tiap kombinasi, dekode **data validasi** lalu hitung WER-nya.

```
contoh:
s_ak=0,5  α=0,8  β=0,5  -> WER 0,158
s_ak=0,5  α=1,0  β=0,5  -> WER 0,152   <- terbaik
s_ak=0,8  α=0,6  β=0,0  -> WER 0,171
... (dan seterusnya)
```

4. Pilih kombinasi dengan WER terendah, lalu **pakai kombinasi itu untuk data uji**.

Kenapa di data validasi, bukan data uji? Supaya tidak curang. Kalau bobot disetel langsung di data uji, hasilnya jadi terlalu bagus dan tidak jujur. Data validasi terpisah dari data uji, jadi bobot yang terpilih benar-benar diuji pada data yang belum pernah dilihat.

## Bobot terbaik di eksperimen kita berapa?

Untuk konfigurasi dua tahap terbaik (Conformer + *spatial attention* + 5-gram + *rescoring* LLaMA-2 7B), kombinasi terbaiknya:

```
s_ak (skala akustik)       = 0,5
α    (bobot LM neural)     = 0,8
β    (bonus penyisipan kata) = 0,5
```

Kombinasi inilah yang dipakai dan menghasilkan WER 0,1556 (irisan willett_4_18).

## Ilustrasi Conformer utuh (tiap kotak di diagram dijelaskan)

Diagram Conformer punya dua bagian: **kolom kiri** = alur *encoder* (bawah ke atas), **kolom kanan** = isi satu blok Conformer (bawah ke atas). (Catatan: diagram ini laju 10 ms; di tugas akhir kita bin-nya 20 ms, tapi alurnya sama. Di versi kita, *spatial attention* dipasang di awal pada 256 kanal mentah, sebelum masuk alur ini.)

Input contoh (laju 10 ms, 8 frame × 2 fitur; aslinya T×256):

```
       f1    f2
t1     1,0   0,0
t2     0,0   1,0
t3     1,0   1,0
t4     0,0   0,0
t5     1,0   0,0
t6     1,0   1,0
t7     0,0   0,0
t8     1,0   0,0
```

### Kolom kiri: alur encoder

**1. SpecAug** (10 ms, hanya saat latih). Menutup (jadikan 0) sebagian potongan waktu dan sebagian fitur secara acak. Misal fitur f2 pada t3-t4 ditutup. Gunanya biar model tidak bergantung pada bagian tertentu, jadi lebih tahan variasi dan tidak mudah *overfitting*. Saat inferensi mati. (Untuk contoh angka di bawah, kita pakai matriks apa adanya seperti saat inferensi.)

**2. Convolution Subsampling** (10 ms -> 40 ms). Konvolusi 2D ber-*stride* yang menggabungkan frame berdekatan, jadi jumlah langkah berkurang. Di sini 4×, jadi 8 frame -> 2 langkah, dan laju berubah 10 ms -> 40 ms. Tiap langkah keluaran merangkum sekitar 4 frame masukan, jadi bawa konteks waktu lebih lebar dan sekuens lebih ringkas (hemat komputasi). Ilustrasi pakai rata-rata tiap grup 4 frame:

```
z1 = rata-rata(t1..t4) = [0,50 , 0,50]
z2 = rata-rata(t5..t8) = [0,75 , 0,25]
```

**3. Linear** (40 ms). Lapisan *linear* yang memetakan tiap langkah ke dimensi model `d`. Contoh `W = [[2,0],[0,2]]`:

```
z1 = [0,50 , 0,50]·W = [1,0 , 1,0]
z2 = [0,75 , 0,25]·W = [1,5 , 0,5]
```

**4. Dropout** (40 ms, hanya saat latih). Acak sebagian nilai dijadikan 0 untuk mencegah *overfitting*. Di contoh ini anggap tidak ada yang tertutup.

**5. Conformer Blocks × N** (40 ms). Barisan `z1, z2` masuk ke N blok Conformer yang ditumpuk. Isi satu blok dijelaskan di kolom kanan. Keluaran blok berukuran sama, jadi masukan blok berikutnya.

### Kolom kanan: isi satu blok Conformer

Urutan (bawah ke atas di diagram): **FFN ½ -> MHSA -> Conv -> FFN ½ -> LayerNorm**. Tanda `+` = koneksi residual (masukan modul dijumlahkan ke keluarannya). Tanda `1/2 x` = residual setengah (khusus dua FFN).

**Modul 1 — Feed Forward (residual setengah).** Isi FFN: Linear (melebar, mis. ×4) -> Swish -> Dropout -> Linear (balik) -> Dropout. Tujuannya memperdalam representasi tiap langkah secara sendiri-sendiri. Rumus dengan residual setengah: `z' = z + 0,5 · FFN(z)`. Contoh, anggap `FFN(z1) = [0,4 , 0,2]`:

```
z1' = [1,0 , 1,0] + 0,5*[0,4 , 0,2] = [1,2 , 1,1]
```

`0,5` bikin pengaruh FFN halus (gaya Macaron, FFN dipecah jadi dua setengah yang mengapit blok).

**Modul 2 — Multi-Head Self Attention (residual penuh).** Isi: LayerNorm -> self-attention banyak *head* (Q, K, V, softmax) -> Dropout. Tiap langkah waktu melihat SEMUA langkah lain, jadi menangkap pola JAUH/global (mis. awal vs akhir ujaran). Mekanismenya persis bagian Transformer di atas. Residual penuh: `z'' = z' + MHSA(z')`. Misal keluarannya (ilustratif) `z1'' = [2,0 , 1,3]`, `z2'' = [1,8 , 0,6]`.

**Modul 3 — Convolution (pembeda Conformer, residual penuh).** Menangkap pola LOKAL antar langkah berdekatan. Masukan dari MHSA: `u1=[2,0 ; 1,3]`, `u2=[1,8 ; 0,6]`. Isi modul berurutan (angka penuh):

**a. LayerNorm** dulu (seperti di section LayerNorm). Untuk d=2 hasilnya selalu pola ±, jadi biar angkanya bermakna kita lanjut dari `u1, u2` langsung.

**b. Pointwise conv (1×1), lebarkan 2 -> 4.** Konvolusi 1×1 = linear per langkah, di sini menggandakan kanal untuk disiapkan ke GLU. `W_pw1 = [[1 , 0,5 , 2 , 0] , [0 , 1 , 0,5 , 1]]`:

```
u1 -> [2,0 , 2,3 , 4,65 , 1,3]
u2 -> [1,8 , 1,5 , 3,9  , 0,6]
```

**c. GLU (gated linear unit).** Belah 4 jadi dua bagian a dan b, keluaran = `a ⊙ sigmoid(b)` (b jadi gerbang 0..1 yang menyeleksi a):

```
u1: a=[2,0 , 2,3]  b=[4,65 , 1,3]  sigmoid(b)=[0,990 , 0,786]  -> g1=[1,980 , 1,808]
u2: a=[1,8 , 1,5]  b=[3,9  , 0,6]  sigmoid(b)=[0,980 , 0,646]  -> g2=[1,764 , 0,969]
```

**d. Depthwise conv.** Konvolusi TIAP kanal sendiri sepanjang waktu, *kernel* 31 (di sini 3 = `[0,25 , 0,5 , 0,25]`, tepi di-*padding*). Ini penangkap pola lokal.

```
kanal 1 sepanjang waktu [1,980 ; 1,764]:
  conv[1] = 0,25*1,980 + 0,5*1,980 + 0,25*1,764 = 1,926
  conv[2] = 0,25*1,980 + 0,5*1,764 + 0,25*1,764 = 1,818
kanal 2 sepanjang waktu [1,808 ; 0,969]:
  conv[1] = 0,25*1,808 + 0,5*1,808 + 0,25*0,969 = 1,598
  conv[2] = 0,25*1,808 + 0,5*0,969 + 0,25*0,969 = 1,179
-> d1=[1,926 , 1,598]   d2=[1,818 , 1,179]
```

**e. BatchNorm -> Swish.** BatchNorm menormalkan tiap kanal antar-contoh (butuh *batch*, dilewati di contoh 1 ujaran ini). Lalu Swish `swish(x)=x·sigmoid(x)`:

```
d1 -> s1 = [1,926*0,873 , 1,598*0,832] = [1,681 , 1,329]
d2 -> s2 = [1,818*0,860 , 1,179*0,765] = [1,564 , 0,902]
```

**f. Pointwise conv (1×1), balik 4 -> 2** (campur antarkanal lagi). `W_pw2 = [[1 , 0,5] , [0,5 , 1]]`:

```
s1 -> p1 = [1,681*1+1,329*0,5 , 1,681*0,5+1,329*1] = [2,346 , 2,170]
s2 -> p2 = [1,564*1+0,902*0,5 , 1,564*0,5+0,902*1] = [2,015 , 1,684]
```

**g. Residual penuh.** Keluaran modul dijumlahkan ke masukannya `u`:

```
z1''' = u1 + p1 = [2,0 , 1,3] + [2,346 , 2,170] = [4,346 , 3,470]
z2''' = u2 + p2 = [1,8 , 0,6] + [2,015 , 1,684] = [3,815 , 2,284]
```

**Modul 4 — Feed Forward (residual setengah).** Sama seperti modul 1, `z'''' = z''' + 0,5 · FFN(z''')`.

**Penutup — LayerNorm.** Normalkan keluaran akhir blok (mean 0, ragam 1 per langkah, lalu diskalakan ulang) biar pelatihan stabil.

### Keluaran akhir

Setelah N blok, keluaran *encoder* (matriks langkah × d, laju 40 ms) dipakai untuk:

- **dua tahap**: diubah jadi probabilitas fonem.
- **E2E**: masuk proyektor -> FM.

Beda utama dengan Transformer polos: ada **modul konvolusi** (pola lokal) dan **dua FFN** yang mengapit (bukan satu). Jadi Conformer dapat pola global (dari MHSA) sekaligus pola lokal (dari konvolusi) sekaligus.

## Conformer encoder di E2E sama dengan yang di dua tahap?

**Tulang punggungnya SAMA**: Conformer + *spatial attention*, dimensi 512, subsampling ~4×. Yang beda cuma kepala keluaran, objektif latih, dan perannya.

| Aspek | Dua tahap | E2E |
| --- | --- | --- |
| Kepala keluaran | probabilitas fonem (41 kelas) | (saat pralatih) probabilitas karakter |
| Objektif latih | CTC level fonem | CTC level karakter (pralatih), lalu ikut dilatih saat E2E |
| Framework | TensorFlow | PyTorch |
| Peran | jadi dekoder fonem (tahap 1) | jadi *encoder*, keluarannya masuk proyektor -> FM |

Jadi arsitektur inti Conformer-nya identik. Bedanya: di dua tahap ia langsung keluar fonem; di E2E ia dipralatih dengan CTC karakter, kepala CTC-nya lalu dibuang, dan *encoder*-nya menyuapi FM (dan ikut di-*fine-tune* saat latih E2E).

## PER Conformer lebih baik tapi WER (tanpa rescoring) kalah dari GRU. Penjelasannya masuk akal?

**Masuk akal, dan sudah dibuktikan angka**, bukan cuma dugaan.

Inti duduk perkaranya:

- **PER cuma lihat fonem teratas tiap frame** (argmax). Conformer lebih sering benar di puncak, jadi PER-nya lebih baik (0,1428 vs 0,1597).
- **Beam search lihat SELURUH bentuk distribusi**, bukan cuma puncaknya. Di sinilah ketajaman berpengaruh.

Kenapa distribusi tajam merugikan beam:

- Distribusi Conformer tajam (entropi 0,0950 vs GRU 0,1448 nats). Artinya model sangat yakin pada satu fonem, dan fonem alternatif dapat peluang nyaris nol.
- Kalau model tajam TAPI salah di suatu frame, fonem yang benar dapat skor nyaris nol, jadi jalur yang benar kena penalti besar dan cepat terpangkas dari beam.
- Distribusi lebih datar (GRU) menyisakan peluang untuk alternatif, jadi lebih banyak jalur bertahan di beam dan jalur benar lebih sering selamat.

Bukti langsung (Tabel IV.5), beam Conformer memang lebih sempit hasilnya:

```
             coverage   oracle WER   rata-rata n-best
Conformer     58,0%      0,1018        32,6
GRU           68,3%      0,0796        48,3
```

`coverage` = seberapa sering transkrip benar ada di daftar n-best. Conformer lebih rendah (58% vs 68%), n-best-nya lebih sedikit, oracle WER-nya lebih tinggi. Persis seperti yang diprediksi.

**Nuansa penting (buat jaga-jaga ditanya):** tajam tidak selalu buruk. Tajam bagus kalau model yakin DAN benar. Masalahnya cuma saat tajam TAPI salah (terlalu percaya diri). Karena itu bilangnya lebih tepat "tajam menyisakan sedikit peluang untuk alternatif" daripada sekadar "memangkas lebih agresif".

**Kenapa setelah rescoring Conformer menang lagi (0,1556 vs GRU 0,1638)?** Untuk ujaran yang jalur benarnya MASIH ada di beam, akurasi fonem Conformer yang lebih baik bikin LLaMA bisa memilih hipotesis yang lebih tepat. Yang hilang dari beam memang tak bisa dipulihkan, tapi secara total keunggulan akurasi Conformer menang.

## LayerNorm input outputnya gimana?

**Input** = satu vektor (satu langkah waktu, panjang `d` fitur). **Output** = vektor ukuran sama, tapi nilainya dinormalkan lalu diskalakan ulang. LayerNorm bekerja **antar fitur di dalam satu langkah** (tiap langkah waktu diproses sendiri-sendiri).

Rumus (untuk satu vektor x):

```
1. mean   μ  = rata-rata semua fitur
2. varians σ² = rata-rata (x - μ)²
3. normal  x̂ = (x - μ) / √(σ² + ε)      (ε kecil biar tak bagi nol)
4. skala   y = γ · x̂ + β                 (γ, β dipelajari, per fitur)
```

Contoh, `x = [1 , 3]` (d=2):

```
μ  = (1+3)/2 = 2
selisih = [-1 , 1] ; kuadrat = [1 , 1] ; σ² = 2/2 = 1 ; √ = 1
x̂ = [(1-2)/1 , (3-2)/1] = [-1 , 1]        <- sudah mean 0, ragam 1
```

Lalu diskalakan pakai γ, β yang dipelajari. Kalau `γ=[1,1]`, `β=[0,0]`, output = `[-1 , 1]`. Kalau `γ=[1,5 ; 0,5]`, `β=[0,2 ; -0,1]`:

```
y = [1,5*(-1)+0,2 , 0,5*1-0,1] = [-1,3 , 0,4]
```

Gunanya: bikin nilai antar fitur seragam skalanya (mean 0, ragam 1) supaya pelatihan stabil, lalu γ dan β memberi model kebebasan menskalakan/menggeser lagi kalau perlu.

**Beda dengan BatchNorm**: LayerNorm normalisasi antar-FITUR dalam satu langkah (tidak butuh *batch*). BatchNorm normalisasi antar-CONTOH dalam satu *batch*. Di Conformer, modul konvolusi pakai BatchNorm, sisanya pakai LayerNorm.

## RTF (real-time factor) dihitung gimana?

Rumus: `RTF = waktu memproses / durasi sinyal`. Membandingkan berapa lama sistem mendekode vs berapa panjang sinyalnya.

Contoh: ujaran punya 150 *bin*, tiap *bin* 20 ms.

```
durasi sinyal   = 150 * 20 ms = 3.000 ms = 3 detik
waktu dekode    = 0,222 detik (misal)
RTF = 0,222 / 3 = 0,074
```

Artinya:

- **RTF < 1**: lebih cepat dari waktu nyata. Sistem selesai mendekode sebelum sinyal sepanjang itu selesai datang, jadi bisa dipakai *real-time*. (0,074 = memproses 3 detik sinyal hanya butuh 0,22 detik, sekitar 13× lebih cepat.)
- **RTF = 1**: pas seimbang.
- **RTF > 1**: lebih lambat dari waktu nyata, tidak bisa mengejar. Misal 3 detik sinyal butuh 6 detik dekode -> RTF = 2.

Catatan: RTF tidak menghitung waktu muat model yang cuma sekali di awal. Di tugas akhir kita semua arsitektur RTF-nya jauh di bawah 1 (paling lambat sekitar 0,32), jadi semuanya sudah lebih cepat dari waktu nyata.

## Kenapa ensembling pakai logistic regression, bukan "pilih confidence tertinggi" saja?

Yang paling sederhana (pilih confidence tertinggi) **sudah dicoba dan gagal**. Ini ablasinya:

```
1a  argmax confidence mentah       0,1700   (kolaps ke E2E, lebih buruk dari 0,1556)
1b  kalibrasi per-model + argmax   0,1465
1c  router logistic regression     0,1441   (headline)
1c  GBDT                           0,1431   (marginal lebih baik)
    oracle best-of-two             0,1089   (langit-langit)
    A dua tahap (tunggal terbaik)  0,1556
    B E2E (tunggal)                0,1711
```

**Kenapa "pilih confidence tertinggi" gagal (1a):** skor kedua sistem beda skala, jadi tak sebanding.

```
E2E       : log-prob per token        ~ -0,3 per token (satu ujaran ~ -3)
Dua tahap : akustik + 5-gram + LLaMA   ~ -64 per ujaran
```

Kalau aturannya "pilih yang lebih dekat 0", E2E (−3) SELALU menang lawan dua tahap (−64). Jadi router bukan memilih cerdas, tapi **selalu ambil E2E** (99,7% ujaran), makanya kolaps.

**Kenapa logistic regression:**

1. **Belajar skala relatif + ambang otomatis**, jadi skor kedua sistem disamakan dulu baru dibandingkan.
2. **Bisa pakai fitur tambahan**, bukan cuma 1 confidence per model. Fitur paling menentukan = entropi beam dua tahap (beam ragu -> rute ke E2E), lalu keyakinan LLaMA per kata. "Pilih confidence tertinggi" tak bisa menangkap ini.
3. Tetap **sederhana, stabil, bisa ditafsirkan** (model linear). Dipilih ketimbang GBDT walau GBDT sedikit lebih baik (0,1431), karena selisihnya dalam rentang derau dan LogReg lebih stabil.

Cerita untuk viva: kita mulai dari yang paling sederhana, itu kolaps karena skor tak sebanding, lalu logistic regression adalah metode paling sederhana yang benar-benar bekerja (samakan skala dulu, baru belajar kapan mempercayai tiap sistem).
