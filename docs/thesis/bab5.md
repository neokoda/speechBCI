# BAB V KESIMPULAN DAN SARAN

Bab ini memaparkan kesimpulan dari penelitian yang telah dilakukan serta saran untuk penelitian dan pengembangan selanjutnya. Bagian V.1 menyajikan kesimpulan yang menjawab apakah tujuan dan ukuran keberhasilan pada bagian I.3 telah tercapai. Bagian V.2 memaparkan saran untuk pengembangan selanjutnya berdasarkan temuan dan keterbatasan penelitian.

## V.1 Kesimpulan

Berdasarkan hasil eksperimen, analisis, dan pembahasan pada bab sebelumnya, dapat ditarik kesimpulan sebagai berikut.

1. Arsitektur dua tahap dengan model berbasis Transformer berhasil dirancang dan diimplementasikan, serta mengungguli kinerja sistem Seto et al. (2025). Tahap pertama berupa dekoder fonem Conformer dengan *spatial attention* mencapai PER 0,1428 yang lebih baik daripada PER dekoder fonem GRU pada penelitian Seto et al. (2025) sebesar sekitar 0,192. Sistem dua tahap utuh dengan tambahan *rescoring* LLaMA-2 7B mencapai WER 0,1556 yang juga lebih baik daripada WER hasil Seto et al. (2025) sebesar 0,169. Dengan demikian, tujuan dan ukuran keberhasilan pertama telah tercapai.

2. Arsitektur *end-to-end* berbasis *Foundation Model* berhasil dirancang dan diimplementasikan. E2E Whisper-large-v3 dengan mekanisme *cross-attention* mencapai WER 0,1716. Nilai ini lebih baik daripada seluruh varian arsitektur dua tahap tanpa *rescoring* model bahasa neural, yaitu Conformer + 5-*gram* dengan WER 0,1858, GRU + 5-*gram* dengan WER 0,1828, dan Transformer + 5-*gram* dengan WER 0,2927. Penambahan *rescoring* LLaMA-2 7B memang menurunkan WER arsitektur dua tahap menjadi 0,1556 sehingga sedikit lebih baik daripada E2E. Akan tetapi, penambahan ini juga meningkatkan kebutuhan penyimpanan dari 44,2 GB menjadi 57,7 GB serta memperlambat dekode beberapa kali lipat. Sebaliknya, E2E Whisper-large-v3 hanya membutuhkan 3,6 GB penyimpanan dan tetap bekerja jauh lebih cepat daripada waktu nyata. Dengan demikian, tujuan dan ukuran keberhasilan kedua telah tercapai karena arsitektur E2E memberikan keseimbangan yang lebih baik antara akurasi, penyimpanan, dan kecepatan inferensi. Kinerja sistem secara menyeluruh tidak hanya diukur dari akurasi semata.

3. Analisis perbandingan antara arsitektur dua tahap dan arsitektur E2E telah dilakukan secara menyeluruh dari sisi akurasi, efisiensi, pola kesalahan, dan keterbatasan. Analisis pada Bab IV menemukan bahwa arsitektur dua tahap dengan *rescoring* LLaMA-2 7B mencapai akurasi tertinggi tetapi memerlukan penyimpanan yang sangat besar, yaitu 57,7 GB. Sebaliknya, arsitektur E2E lebih ringan, yaitu 2,0 hingga 9,9 GB, dengan akurasi yang kompetitif. Analisis lanjutan juga menemukan bahwa kedua arsitektur saling melengkapi pada 22,8% ujaran uji dan bahwa pendekatan dua tahap memiliki *ceiling* teoretis yang membatasi peningkatan akurasi melalui *rescoring*. Dengan demikian, tujuan dan ukuran keberhasilan ketiga telah tercapai.

## V.2 Saran

Berdasarkan temuan dan keterbatasan penelitian ini, beberapa saran untuk penelitian dan pengembangan selanjutnya dipaparkan sebagai berikut.

1. **Penambahan data latih dan pralatih *encoder* ECoG.** Pembahasan pada Bab IV menunjukkan kinerja arsitektur E2E lebih dibatasi oleh ketersediaan data daripada oleh kapasitas model. Penambahan jumlah perekaman ECoG atau pemanfaatan teknik *transfer learning* dari *encoder* yang sudah dilatih pada data neural lintas-subjek berpotensi meningkatkan kinerja arsitektur E2E dan mengejar akurasi arsitektur dua tahap.

2. **Pengembangan *rescorer* yang lebih baik untuk pipeline dua tahap.** Analisis pada Bab IV menunjukkan *oracle WER* dari daftar *n-best* WFST 5-*gram* hanya 0,1018, sedangkan WER aktual setelah *rescoring* LLaMA-2 7B masih sebesar 0,1556. Selisih ini menunjukkan *rescorer* LLaMA-2 7B belum sepenuhnya memanfaatkan ruang hipotesis yang tersedia. Pengembangan *rescorer* yang lebih sesuai dengan distribusi hipotesis dari pipeline atau perluasan daftar *n-best* berpotensi menutup selisih tersebut.

3. **Eksplorasi *ensembling* antara arsitektur E2E dan dua tahap.** Analisis tumpang tindih kesalahan pada Bab IV menunjukkan kedua arsitektur menangkap pola kesalahan yang berbeda. Jika untuk setiap ujaran dipilih arsitektur dengan WER lebih rendah, WER gabungannya turun menjadi 0,1249, jauh di bawah WER tunggal masing-masing arsitektur. Mekanisme *ensembling* yang dapat menyatukan kekuatan kedua arsitektur secara dinamis berpotensi mengangkat akurasi melebihi tiap arsitektur tunggal.

4. **Pengembangan arah *speaker-independent*.** Sesuai batasan pada Bab I, penelitian ini dilakukan pada satu partisipan tunggal sehingga model belum dapat langsung dipakai pada partisipan lain. Pengembangan teknik adaptasi domain, pralatih pada data multipartisipan, atau pemanfaatan model fondasi neural lintas-subjek dapat menjadi langkah penting menuju sistem yang siap diterapkan secara klinis kepada berbagai pengguna.

5. **Perluasan kosakata dan pengujian pada dataset lain.** Sesuai batasan pada Bab I, penelitian ini terbatas pada kosakata yang tersedia dalam dataset Willett et al. (2023). Pengujian arsitektur pada dataset BCI lain dengan kosakata atau domain yang berbeda akan memperluas pemahaman tentang generalisasi arsitektur, khususnya kemampuan *Foundation Model* pada arsitektur E2E untuk memanfaatkan pengetahuan linguistik bawaannya pada kosakata terbuka.
