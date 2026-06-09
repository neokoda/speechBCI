# Daftar Istilah dan Aturan Penulisan Huruf Miring

Acuan: PUEBI/EYD (Permendikbud) dan KBBI.

## Aturan ringkas

Berdasarkan EYD, huruf miring dipakai untuk **kata atau ungkapan bahasa asing yang belum diserap** ke bahasa Indonesia. Maka:

**Dimiringkan** — istilah asing yang belum diserap/dibakukan di KBBI (mis. *self-attention*, *beam search*, *end-to-end*).

**TIDAK dimiringkan:**
1. **Kata serapan baku** yang sudah ada di KBBI — fonem, elektroda, neuron, frekuensi, matriks, parameter, korpus, akustik, linguistik, konvolusi, normalisasi, inferensi, modalitas, token.
2. **Nama diri (nama produk/model/arsitektur)** — Transformer, Conformer, Whisper, Qwen, Cohere, Canary-Qwen, Granite-Speech, LLaMA-2, LLaVA, Adam, AdamW. Nama diri mengikuti aturan kapital, bukan huruf miring.
3. **Singkatan/akronim** — BCI, ECoG, EEG, fMRI, CTC, WFST, RNN, BiRNN, GRU, LoRA, FM, WER, PER, CER, RTF, WPM, ALS, SNR, FFN, GLU. Kepanjangan berbahasa asingnya dimiringkan saat **pertama** disebut, akronimnya sendiri tidak (mis. *connectionist temporal classification* (CTC), selanjutnya CTC).

### Jawaban untuk "Transformer"
**Transformer adalah nama arsitektur (nama diri), bukan kata asing umum.** Maka **tidak dimiringkan** dan diawali huruf kapital. Hal yang sama berlaku untuk Conformer, Whisper, LLaMA, Qwen, Cohere, dan LLaVA. Yang dimiringkan adalah istilah teknik umum di sekitarnya, seperti *self-attention*, *cross-attention*, dan *end-to-end*.

### Catatan konsistensi
- Umumnya cukup dimiringkan pada **kemunculan pertama**. Sebagian pembimbing/template ITB meminta konsisten di seluruh dokumen. Sesuaikan dengan arahan pembimbing.
- *Foundation Model* dimiringkan karena merupakan **istilah kelas model** (bukan nama produk tunggal).

---

## Tabel istilah

### A. Dimiringkan (istilah asing belum diserap)

| Istilah | Keterangan |
| --- | --- |
| *neuroprosthesis* | prostesis saraf untuk bicara |
| *anarthria* | ketidakmampuan mengartikulasikan bicara |
| *locked-in* | kondisi sadar penuh tetapi nyaris lumpuh total |
| *quadriplegia* | kelumpuhan empat anggota gerak |
| *end-to-end* (E2E) | pemetaan langsung input ke teks tanpa tahap perantara |
| *self-attention* | atensi internal antartoken |
| *cross-attention* | atensi silang teks ke memori ECoG |
| *spatial attention* | atensi pada dimensi elektroda |
| *beam search* | pencarian jalur dengan lebar terbatas |
| *shallow fusion* | penggabungan skor akustik dan model bahasa |
| *rescoring* | penilaian ulang daftar hipotesis |
| *n-best* | daftar n hipotesis terbaik |
| *threshold crossings* | fitur pelintasan ambang |
| *spike band power* | daya pita *spike* |
| *subsampling* | penurunan laju cuplik pada dimensi waktu |
| *dropout* | regularisasi dengan menonaktifkan unit acak |
| *learning rate* | laju pembelajaran |
| *warmup* | pemanasan laju pembelajaran |
| *batch* | ukuran kelompok data pelatihan |
| *weight decay* | peluruhan bobot |
| *layer normalization* | normalisasi lapisan |
| *feed-forward* (FFN) | lapisan umpan maju |
| *positional encoding* | penyandian posisi |
| *softmax*, *softsign* | fungsi aktivasi/normalisasi |
| *baseline* | pembanding dasar |
| *bottleneck* | sumbatan/leher botol kinerja |
| *coverage* | fraksi referensi yang termuat di *n-best* |
| *oracle WER* | batas bawah WER bila selalu memilih hipotesis terbaik |
| *overfitting* | model terlalu menyesuaikan data latih |
| *teacher forcing* | pelatihan dengan token rujukan |
| *mouthing* | menggerakkan mulut tanpa bersuara |
| *eye tracking* | pelacakan gerak mata |
| *encoder*, *decoder* | pengode/pendekode |
| *encoder-only*, *decoder-only*, *encoder-decoder* | varian arsitektur |
| *Foundation Model* (FM) | kelas model fondasi terlatih skala besar |
| *fine-tuning*, *full fine-tuning* | penyetelan lanjut |
| *hyperparameter tuning* | penyetelan hiperparameter |
| *grid search* | pencarian kombinasi nilai |
| *text shortcut* | jalan pintas memprediksi teks dari teks |
| *word insertion bonus* | bonus penyisipan kata |
| *real-time*, *wall-clock* | waktu nyata |
| *downstream tasks* | tugas hilir |
| *transfer learning* | pembelajaran transfer |
| *kernel*, *head*, *hidden* | istilah komponen model |

### B. TIDAK dimiringkan — nama diri (model/arsitektur/algoritma)

| Istilah | Jenis |
| --- | --- |
| Transformer | nama arsitektur |
| Conformer | nama arsitektur |
| Whisper-medium.en, Whisper-large-v3 | nama model |
| Qwen (Qwen3.5-0.8B, Qwen3-1.7B) | nama model |
| Cohere Transcribe | nama model |
| Canary-Qwen-2.5B | nama model |
| Granite-Speech-4.1-2B | nama model |
| LLaMA-2 | nama model |
| LLaVA | nama model/gaya adaptasi |
| Adam, AdamW | nama *optimizer* |
| Swish, SiLU | nama fungsi aktivasi |

### C. TIDAK dimiringkan — singkatan/akronim

| Singkatan | Kepanjangan (miring saat pertama disebut) |
| --- | --- |
| BCI | *brain-computer interface* |
| ECoG | elektrokortikografi |
| EEG | elektroensefalografi |
| fMRI | *functional magnetic resonance imaging* |
| CTC | *connectionist temporal classification* |
| WFST | *weighted finite-state transducer* |
| RNN / BiRNN | *(bidirectional) recurrent neural network* |
| GRU | *gated recurrent unit* |
| LoRA | *low-rank adaptation* |
| FM | *Foundation Model* |
| WER / PER / CER | *word/phoneme/character error rate* |
| RTF | *real-time factor* |
| WPM | kata per menit (*words per minute*) |
| ALS | *amyotrophic lateral sclerosis* |
| SNR | rasio sinyal-terhadap-derau |

### D. TIDAK dimiringkan — kata serapan baku (KBBI)

fonem, elektroda, neuron, frekuensi, matriks, parameter, korpus, akustik, linguistik, konvolusi, normalisasi, inferensi, modalitas, token, autoregresif, augmentasi, hipotesis, validasi.
