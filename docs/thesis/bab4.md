# BAB IV EVALUASI

Bab ini menjelaskan prosedur yang dijalankan untuk mengevaluasi solusi yang dirancang pada Bab III beserta hasil dan pembahasannya. Bagian IV.1 memaparkan lingkungan pengembangan tempat seluruh eksperimen dijalankan. Bagian IV.2 menjabarkan desain eksperimen yang dilakukan. Bagian IV.3 memaparkan hasil setiap eksperimen. Terakhir, bagian IV.4 membahas dan menganalisis hasil tersebut.

## IV.1 Lingkungan Pengembangan

Seluruh proses pelatihan dan evaluasi model dijalankan pada GPU *cloud* yang disewa melalui layanan RunPod. GPU yang dipilih adalah NVIDIA GeForce RTX 4090. Pilihan ini didasari oleh ketersediaan, biaya sewa yang wajar, serta kapasitas memori 24 GB yang mencukupi untuk melatih *Foundation Model* berukuran besar dengan teknik LoRA. Spesifikasi lengkap lingkungan pengembangan ditunjukkan pada Tabel IV.1.

**Tabel IV.1** Spesifikasi lingkungan pengembangan.

| Komponen           | Spesifikasi                                                                         |
| ------------------ | ----------------------------------------------------------------------------------- |
| Prosesor           | 16 vCPU (host AMD EPYC)                                                             |
| RAM                | 62 GB                                                                               |
| GPU                | NVIDIA GeForce RTX 4090, 24 GB GDDR6X, arsitektur Ada Lovelace, 16.384*CUDA core* |
| Penyimpanan        | 200 GB                                                                              |
| Sistem Operasi     | Ubuntu 22.04 LTS                                                                    |
| Lingkungan virtual | Python*venv* berisi PyTorch (CUDA 12) dan TensorFlow 2.15                         |

Lingkungan virtual dibagi sesuai kebutuhan tiap jalur. Jalur *end-to-end* (E2E) dan *rescoring* model bahasa neural menggunakan PyTorch beserta pustaka transformers dan peft. Sementara itu, dekoder fonem dan dekode WFST pada arsitektur dua tahap menggunakan TensorFlow. Metrik kesalahan dihitung dengan pustaka jiwer.

## IV.2 Desain Eksperimen

Terdapat dua eksperimen yang dilakukan pada tugas akhir ini. Eksperimen pertama menguji kinerja dekoder fonem sebagai tahap pertama arsitektur dua tahap. Eksperimen kedua membandingkan kinerja arsitektur secara utuh, baik arsitektur dua tahap maupun arsitektur E2E, dari sisi akurasi dan efisiensi.

### IV.2.1 Eksperimen Dekode Fonem

Eksperimen ini bertujuan membandingkan kinerja empat dekoder fonem dalam memetakan fitur neural menjadi probabilitas fonem. Keempat dekoder tersebut adalah GRU sebagai *baseline*, Transformer murni, Conformer biasa, dan Conformer dengan *spatial attention*. GRU dipilih sebagai *baseline* karena merupakan dekoder yang digunakan oleh Willett et al. (2023). Transformer murni, Conformer murni, dan Conformer dengan *spatial attention* dipilih untuk menguji apakah arsitektur berbasis Transformer dapat melampaui *baseline* tersebut. Metrik yang digunakan adalah *phoneme error rate* (PER).

Setiap dekoder memiliki hiperparameter spesifik sebagai berikut.

1. **GRU.** Model terdiri atas 5 lapisan GRU searah dengan 1024 unit per lapisan dan berjumlah sekitar 53,6 juta parameter. Fitur masukan terlebih dahulu melewati lapisan input berdimensi 256 dengan fungsi aktivasi *softsign* dan *dropout* sebesar 0,4. Masukan juga ditumpuk pada dimensi waktu untuk melakukan *subsampling* dengan faktor empat sebelum diproses lapisan GRU.
2. **Transformer murni.** Model memiliki dimensi *hidden* 512, 4 lapisan, 8 *attention head*, dimensi lapisan *feed-forward* 2048, dan *dropout* sebesar 0,1. Informasi posisi dibubuhkan dengan *positional encoding* sinusoidal. Berbeda dengan dekoder lain, Transformer murni dilatih dengan *learning rate* yang lebih kecil, yaitu 0,015, untuk menjaga kestabilan pelatihan.
3. **Conformer vanila.** Model memiliki dimensi *hidden* 512, 4 lapisan, 8 *attention head*, dimensi lapisan *feed-forward* 2048, *kernel* konvolusi berukuran 31, dan *dropout* sebesar 0,1. Informasi posisi disandikan dengan *positional encoding* sinusoidal.
4. **Conformer dengan *spatial attention*.** Model memiliki konfigurasi yang sama dengan Conformer vanila, tetapi ditambah modul *spatial attention* pada dimensi elektroda dengan dimensi 64 dan 4 *head*.

Keempat dekoder dilatih dengan fungsi *loss* CTC, *optimizer* Adam, dan ukuran *batch* 32. Laju pembelajaran mengikuti jadwal *cosine* dengan *warmup* seperti yang dijelaskan pada Bab III. Pelatihan menggunakan *early stopping* berdasarkan PER validasi, dan bobot model dengan PER validasi terendah disimpan sebagai hasil akhir.

### IV.2.2 Eksperimen Evaluasi Arsitektur

Eksperimen ini bertujuan membandingkan kinerja arsitektur secara utuh dari sisi akurasi dan efisiensi. Akurasi diukur dengan *word error rate* (WER) dan *character error rate* (CER), sedangkan efisiensi diukur dengan ukuran penyimpanan, *real-time factor* (RTF), dan kata per menit (WPM). Arsitektur yang dibandingkan adalah arsitektur dua tahap dan arsitektur E2E dengan berbagai *Foundation Model* yang telah dijelaskan pada Bab III. Daftar arsitektur yang dievaluasi adalah sebagai berikut.

1. **Dua tahap (GRU + 5-*gram*).** Dekoder fonem GRU yang dilanjutkan dekode model bahasa 5-*gram* berbasis WFST.
2. **Dua tahap (Transformer + 5-*gram*).** Dekoder fonem Transformer murni yang dilanjutkan dekode model bahasa 5-*gram* berbasis WFST.
3. **Dua tahap (Conformer + 5-*gram*).** Dekoder fonem Conformer dengan *spatial attention* yang dilanjutkan dekode model bahasa 5-*gram* berbasis WFST.
4. **Dua tahap (GRU + 5-*gram* + LLaMA-2 7B).** Daftar *n-best* dari dekode WFST pada dekoder GRU di-*rescore* dengan model bahasa neural LLaMA-2 7B (Touvron et al., 2023).
5. **Dua tahap (Conformer + 5-*gram* + LLaMA-2 7B).** Daftar *n-best* dari dekode WFST pada dekoder Conformer dengan *spatial attention* di-*rescore* dengan model bahasa neural LLaMA-2 7B.
6. **E2E Qwen (gaya *LLaVA*).** Arsitektur E2E yang memanfaatkan model bahasa teks Qwen3.5-0.8B dengan adaptasi gaya *LLaVA*.
7. **E2E Whisper-medium.en (*cross-attention*).** Arsitektur E2E yang memanfaatkan model audio Whisper-medium.en melalui *cross-attention*.
8. **E2E Whisper-large-v3 (*cross-attention*).** Arsitektur E2E yang memanfaatkan model audio Whisper-large-v3 melalui *cross-attention*.
9. **E2E Cohere Transcribe (*cross-attention*).** Arsitektur E2E yang memanfaatkan model audio Cohere Transcribe melalui *cross-attention*.
10. **E2E Canary-Qwen (gaya *LLaVA*).** Arsitektur E2E yang menggunakan ulang model bahasa Qwen3-1.7B dari Canary-Qwen-2.5B dengan adaptasi gaya *LLaVA*.
11. **E2E Granite-Speech (gaya *LLaVA*).** Arsitektur E2E yang menggunakan ulang model bahasa dari Granite-Speech-4.1-2B dengan adaptasi gaya *LLaVA*.

WER dan CER dihitung pada tingkat korpus (*micro-average*), yaitu jumlah seluruh kesalahan dibagi jumlah seluruh kata atau karakter pada data uji. Sama seperti eksperimen dekode fonem, akurasi dilaporkan pada data uji dengan sesi perekaman yang sebanding dengan laporan Willett et al. (2023).

## IV.3 Hasil Eksperimen

### IV.3.1 Hasil Dekode Fonem

Hasil PER keempat dekoder fonem ditunjukkan pada Tabel IV.2. Conformer dengan *spatial attention* mencapai PER terendah, yaitu 0,1428. Conformer vanila menempati posisi kedua. GRU *baseline* berada di atas kedua varian Conformer tersebut, sedangkan Transformer murni memiliki PER tertinggi.

**Tabel IV.2** Hasil PER empat dekoder fonem.

| Dekoder Fonem                               | PER              |
| ------------------------------------------- | ---------------- |
| GRU 1024 unit 5 lapisan                     | 0,1597           |
| Transformer murni                           | 0,2444           |
| Conformer vanila                            | 0,1477           |
| **Conformer + *spatial attention*** | **0,1428** |

Hasil ini menunjukkan kedua varian Conformer mengungguli GRU *baseline*, sedangkan Transformer murni justru berada di bawah GRU. Hal ini menunjukkan mekanisme *self-attention* murni belum cukup untuk dekode fonem dari sinyal ECoG dan modul konvolusi pada Conformer berperan penting dalam menangkap pola lokal sinyal. Conformer dengan *spatial attention* juga mengungguli dekoder fonem GRU pada penelitian Seto (2025) yang memiliki PER sekitar 0,192. Dengan demikian, dekoder fonem berbasis Conformer terbukti lebih baik dalam memetakan fitur neural menjadi probabilitas fonem.

### IV.3.2 Hasil Evaluasi Arsitektur

Hasil akurasi setiap arsitektur ditunjukkan pada Tabel IV.3. Arsitektur dua tahap dengan *rescoring* LLaMA-2 7B mencapai WER terendah, yaitu 0,1556. Di antara arsitektur E2E, Whisper-large-v3 mencapai akurasi tertinggi dengan WER 0,1716.

**Tabel IV.3** Hasil WER dan CER setiap arsitektur.

| Arsitektur                                                | WER              | CER              |
| --------------------------------------------------------- | ---------------- | ---------------- |
| Dua tahap (GRU + 5-*gram*)                              | 0,1828           | 0,1327           |
| Dua tahap (Transformer + 5-*gram*)                      | 0,2927           | (menyusul)       |
| Dua tahap (Conformer + 5-*gram*)                        | 0,1858           | 0,1253           |
| Dua tahap (GRU + 5-*gram* + LLaMA-2 7B)                 | 0,1638           | 0,1194           |
| **Dua tahap (Conformer + 5-*gram* + LLaMA-2 7B)** | **0,1556** | **0,1127** |
| E2E Qwen (gaya*LLaVA*)                                  | 0,2537           | 0,2413           |
| E2E Whisper-medium.en                                     | 0,1760           | 0,1508           |
| **E2E Whisper-large-v3**                            | **0,1716** | **0,1428** |
| E2E Cohere Transcribe                                     | 0,1776           | 0,1523           |
| E2E Canary-Qwen                                           | (menyusul)       | (menyusul)       |
| E2E Granite-Speech                                        | (menyusul)       | (menyusul)       |

Hasil efisiensi setiap arsitektur ditunjukkan pada Tabel IV.4. Seluruh arsitektur, baik dua tahap maupun E2E, memiliki RTF jauh di bawah satu sehingga semuanya mampu bekerja lebih cepat daripada durasi ucapan aslinya. Dari sisi penyimpanan, arsitektur E2E jauh lebih ringan, yaitu 2,0 hingga 9,9 GB, dibandingkan arsitektur dua tahap yang mencapai 44,2 hingga 57,8 GB.

**Tabel IV.4** Jumlah parameter, ukuran penyimpanan, dan kecepatan setiap arsitektur.

| Arsitektur                                      | Total Parameter (komponen terbesar)     | Penyimpanan Total (komponen terbesar) | RTF        | WPM        |
| ----------------------------------------------- | --------------------------------------- | ------------------------------------- | ---------- | ---------- |
| Dua tahap (GRU + 5-*gram*)                    | 53,6 juta (GRU fonem 53,55 juta)        | 44,3 GB (TLG.fst 44,1 GB)             | 0,0155     | 4.556      |
| Dua tahap (Transformer + 5-*gram*)            | (menyusul)                              | (menyusul)                            | (menyusul) | (menyusul) |
| Dua tahap (Conformer + 5-*gram*)              | 28,5 juta (Conformer fonem 28,49 juta)  | 44,2 GB (TLG.fst 44,1 GB)             | 0,0069     | 10.307     |
| Dua tahap (GRU + 5-*gram* + LLaMA-2 7B)       | 6,79 miliar (LLaMA-2 6,74 miliar)       | 57,8 GB (TLG.fst 44,1 GB)             | 0,0756     | 935        |
| Dua tahap (Conformer + 5-*gram* + LLaMA-2 7B) | 6,77 miliar (LLaMA-2 6,74 miliar)       | 57,7 GB (TLG.fst 44,1 GB)             | 0,0430     | 1.650      |
| E2E Qwen (gaya*LLaVA*)                        | ~0,86 miliar (FM 0,8 miliar)            | 2,1 GB (FM 1,77 GB)                   | 0,0710     | 895        |
| E2E Whisper-medium.en                           | ~0,80 miliar (FM 769 juta)              | 2,0 GB (FM 1,54 GB)                   | 0,0455     | 1.394      |
| E2E Whisper-large-v3                            | ~1,57 miliar (FM 1,54 miliar)           | 3,6 GB (FM 3,09 GB)                   | 0,0742     | 857        |
| E2E Cohere Transcribe                           | ~2,0 miliar (FM ~2 miliar)              | 4,5 GB (FM 4,13 GB)                   | 0,0206     | 3.091      |
| E2E Canary-Qwen                                 | ~4,2 miliar (encoder Canary 2,5 miliar) | 9,9 GB (encoder Canary 5,12 GB)       | 0,1260     | 503        |
| E2E Granite-Speech                              | ~2,0 miliar (FM ~2 miliar)              | 5,3 GB (FM 4,87 GB)                   | 0,0672     | 943        |

Nilai RTF pada Tabel IV.4 adalah perbandingan waktu dekode terhadap durasi sinyal neural, sedangkan WPM adalah jumlah kata yang dihasilkan per menit waktu nyata (*wall-clock*) sebagai ukuran laju keluaran. RTF dan WPM tidak menghitung waktu muat model satu kali. Jumlah parameter sebagian *Foundation Model* bersifat perkiraan dari ukuran *snapshot* fp16, sedangkan ukuran penyimpanan dilaporkan sebagai bobot siap pakai. Pada model 5-*gram*, berkas TLG.fst tidak memiliki parameter terlatih (non-parametrik), tetapi tetap mendominasi penyimpanan sebesar 44,1 GB.

## IV.4 Pembahasan

Bagian ini membahas dan menganalisis hasil kedua eksperimen.

Di antara seluruh arsitektur, E2E Whisper-large-v3 memberikan keseimbangan terbaik antara akurasi dan efisiensi. Dari sisi akurasi, model ini mencapai WER 0,1716 dan hanya kalah dari arsitektur dua tahap yang memakai *rescoring* LLaMA-2 7B. Apabila bantuan model bahasa neural tersebut ditiadakan, seluruh arsitektur dua tahap justru kalah dari Whisper-large-v3, yaitu GRU dengan WER 0,1828, Conformer dengan WER 0,1858, dan Transformer dengan WER 0,2927. Dari sisi efisiensi, Whisper-large-v3 hanya membutuhkan penyimpanan 3,6 GB yang  jauh lebih kecil daripada arsitektur dua tahap mana pun. Oleh karena itu, Whisper-large-v3 menjadi pilihan paling seimbang untuk penerapan dalam dunia nyata.

Arsitektur dua tahap dengan *rescoring* LLaMA-2 7B memang mencapai WER dan CER terendah secara keseluruhan, yaitu WER 0,1556. Nilai ini mengungguli *baseline* Willett et al. (2023) yang memiliki WER sekitar 0,174 dan hasil Seto (2025) yang memiliki WER sekitar 0,169. Akan tetapi, keunggulan akurasi ini disertai beban penyimpanan dan komputasi yang besar. Penyimpanannya mencapai 57,7 GB yang didominasi oleh berkas TLG.fst sebesar 44,1 GB dan bobot LLaMA-2 7B sebesar 13,5 GB. Penambahan *rescoring* LLaMA-2 juga memperlambat dekode hingga beberapa kali lipat, terlihat dari RTF dekode GRU yang naik dari 0,0155 menjadi 0,0756. Dengan demikian, peningkatan akurasi pada pendekatan dua tahap ditukar dengan kebutuhan penyimpanan dan latensi yang jauh lebih tinggi.

Dari sisi kecepatan, seluruh arsitektur sebenarnya sudah bekerja lebih cepat daripada waktu nyata karena RTF-nya jauh di bawah satu, bahkan untuk arsitektur yang paling lambat sekalipun. Oleh karena itu, pertimbangan utama bukan lagi kecepatan, melainkan penyimpanan dan akurasi. Pada aspek penyimpanan, arsitektur E2E unggul jauh dengan kebutuhan 2,0 hingga 9,9 GB dibandingkan 44,2 hingga 57,8 GB pada arsitektur dua tahap. Beban penyimpanan dua tahap ini hampir seluruhnya berasal dari berkas TLG.fst model 5-*gram* sebesar 44,1 GB, sedangkan dekoder fonemnya sendiri hanya berukuran ratusan *megabyte*.

Di antara arsitektur E2E, model audio yang diadaptasi melalui *cross-attention* secara konsisten mengungguli model bahasa teks yang diadaptasi gaya *LLaVA*. Whisper-large-v3, Whisper-medium.en, dan Cohere Transcribe seluruhnya memiliki WER yang jauh lebih rendah daripada Qwen gaya *LLaVA*. Hasil ini sejalan dengan rancangan pada Bab III, yaitu pendekatan *cross-attention* untuk mengatasi *text shortcut* yang muncul pada pendekatan *LLaVA*. Selain itu, model yang sudah dilatih pada *modality* audio lebih cocok untuk memetakan sinyal ECoG menjadi teks dibandingkan model bahasa teks murni. Di antara model audio, ukuran model yang lebih besar cenderung memberikan akurasi yang lebih baik. Hal ini terlihat dari Whisper-large-v3 yang mengungguli Whisper-medium.en.

Hasil kecepatan antararsitektur E2E juga menunjukkan pola yang menarik. Cohere Transcribe menjadi model *cross-attention* tercepat dengan RTF 0,0206 meskipun jumlah parameternya paling besar. Hal ini terjadi karena latensi dekode ditentukan oleh kedalaman dekoder, bukan oleh total parameter. Pada jalur *cross-attention*, *encoder* audio bawaan FM tidak dijalankan karena keluaran *encoder* ECoG langsung dijadikan memori *cross-attention* sehingga hanya dekoder yang berjalan secara autoregresif. Dekoder Cohere hanya memiliki 8 lapisan, sedangkan dekoder Whisper-large-v3 memiliki 32 lapisan. Sebagian besar parameter Cohere berada pada *encoder* audionya yang justru tidak dipakai pada pendekatan ini.

Pada eksperimen dekode fonem, penambahan *spatial attention* menurunkan PER dibandingkan Conformer vanila. Hasil ini menunjukkan modul *spatial attention* membantu model menangkap dependensi antarelektroda pada sinyal ECoG yang terdiri atas banyak kanal. Kedua varian Conformer juga mengungguli GRU *baseline*, sedangkan Transformer murni memiliki PER tertinggi. Oleh karena itu, modul konvolusi pada Conformer berperan penting dan arsitektur Transformer murni belum cukup untuk dekode fonem dari sinyal ECoG.

Terdapat temuan menarik pada hubungan antara PER dekoder fonem dan WER akhir arsitektur dua tahap. Meskipun Conformer dengan *spatial attention* memiliki PER yang lebih baik daripada GRU, dekoder GRU justru menghasilkan WER yang sedikit lebih rendah pada dekode 5-*gram*, yaitu 0,1828 dibandingkan 0,1858. Hal ini disebabkan distribusi probabilitas fonem dari Conformer cenderung lebih tajam sehingga *beam search* lebih mudah memangkas jalur yang sebenarnya benar. Sebaliknya, distribusi GRU yang lebih landai mempertahankan lebih banyak hipotesis sehingga peluang menemukan jalur yang benar lebih besar. Ketajaman distribusi Conformer ini sekaligus membuat dekode WFST-nya sekitar empat kali lebih cepat daripada GRU karena entropi yang lebih rendah memungkinkan *beam search* memangkas lebih agresif. Dengan demikian, ketajaman distribusi merupakan pedang bermata dua, yaitu mempercepat dekode tetapi berisiko memangkas jalur yang benar. Meskipun begitu, keunggulan WER GRU tersebut hilang setelah *rescoring* dengan LLaMA-2 7B. Conformer dengan *spatial attention* kembali unggul dengan WER 0,1556 dibandingkan 0,1638 milik GRU karena model bahasa neural mampu memilih hipotesis yang lebih baik dari daftar *n-best*.

Arsitektur E2E pada tugas akhir ini cenderung mengalami *overfitting* karena jumlah data latih yang terbatas. Beberapa percobaan lanjutan dengan pelatihan yang lebih panjang maupun penambahan parameter tidak mampu menurunkan WER arsitektur E2E secara berarti. Hal ini menunjukkan kinerja arsitektur E2E lebih dibatasi oleh ketersediaan data daripada oleh kapasitas model. Penambahan data latih atau pralatih *encoder* pada data ucapan eksternal berpotensi meningkatkan kinerja arsitektur E2E dan dapat menjadi arah pengembangan selanjutnya.
