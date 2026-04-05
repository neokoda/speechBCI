#!/usr/bin/env python3
"""
End-to-End Language Model Pipeline Evaluation
==============================================
Evaluates the conformer_spatial_24sess neural decoder integrated with
GPT-2 (124M) and Gemma 3 (270M) language models via N-best rescoring
(shallow fusion), following Seto et al. (2025) methodology.

Pipeline:
  1. Load Conformer checkpoint → run inference → get logits + ground truth
  2. CTC prefix beam search → N-best phoneme sequences with acoustic scores
  3. Phoneme sequences → word sequences via CMU pronunciation dict DP
  4. LLM rescore: total = acoustic_scale * ac_score + lm_weight * lm_score
  5. Compute CER, WER, PER, WPM across all test utterances

Usage:
    python AnalysisExamples/eval_lm_pipeline.py [--lm gpt2|gemma|both]
"""

import os
import sys
import json
import time
import argparse
import logging
from collections import defaultdict

import numpy as np

# ── GPU / TF setup ─────────────────────────────────────────────────────────
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import site
# Add NVIDIA pip-installed CUDA libs to LD_LIBRARY_PATH so TF finds the GPU.
# Works whether running from the venv311 (TF 2.15) or system Python (TF 2.21+).
_NV_BASE = os.path.join(site.getsitepackages()[0], 'nvidia')
if os.path.isdir(_NV_BASE):
    _NV_SUBDIRS = ['cudnn', 'cublas', 'cuda_nvrtc', 'cuda_runtime',
                   'cufft', 'cusolver', 'cusparse', 'nvjitlink']
    _NV_LIBS = ':'.join(
        os.path.join(_NV_BASE, d, 'lib')
        for d in _NV_SUBDIRS
        if os.path.isdir(os.path.join(_NV_BASE, d, 'lib'))
    )
    if _NV_LIBS:
        os.environ['LD_LIBRARY_PATH'] = _NV_LIBS + ':' + os.environ.get('LD_LIBRARY_PATH', '')

import tensorflow as tf

# Enable GPU memory growth so TF doesn't pre-allocate the entire GPU,
# leaving room for PyTorch LM scoring on the same device.
for _gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu, True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'NeuralDecoder'))
from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder
from neuralDecoder.datasets.speechDataset import PHONE_DEF_SIL
from neuralDecoder.utils.rnnEval import wer as edit_distance
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
CKPT_DIR       = '/workspace/speechBCI/experiments/24sess/conformer_spatial_24sess'
DATA_DIR       = '/workspace/speechBCI/data/derived/tfRecords'
OUTPUT_DIR     = '/workspace/speechBCI/experiments/lm_pipeline'
GPT2_MODEL_ID  = 'gpt2'
GEMMA_MODEL_ID = 'google/gemma-3-270m'

# Each output frame covers strides * 20ms = 80ms of audio
FRAME_DURATION_SEC = 0.080   # seconds per output logit step

# ── Phoneme definitions ──────────────────────────────────────────────────────
# PHONE_DEF_SIL = [39 phonemes + 'SIL']; blank = class 40
# Raw model logits layout: [class0..38=phonemes, class39=SIL, class40=blank]
BLANK_IDX = 40
SIL_IDX   = 39
PHONE_DEF = PHONE_DEF_SIL[:-1]   # 39 phonemes, no SIL
IDX_TO_PHONE = {i: p for i, p in enumerate(PHONE_DEF_SIL)}

# ═══════════════════════════════════════════════════════════════════════════
# 1.  NEURAL DECODER INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def _fix_all_checkpoint_compat(model, input_layers, ckpt_path):
    """Comprehensive TF 2.15 → 2.21 checkpoint compatibility fix.

    In TF 2.21 / Keras 3, several weight tracking paths changed from Keras 2:

    1. EinsumDense (used inside MultiHeadAttention) renames kernel attribute:
         OLD: .../_query_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
         NEW: .../_query_dense/_kernel/.ATTRIBUTES/VARIABLE_VALUE

    2. Direct Dense layers on model (input_proj, output dense) also rename:
         OLD: net/input_proj/kernel/.ATTRIBUTES/VARIABLE_VALUE
         NEW: net/input_proj/_kernel/.ATTRIBUTES/VARIABLE_VALUE

    3. Sequential inputLayer models changed internal tracking structure:
         OLD: inputLayer_N/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
         NEW: inputLayer_N/_functional/_operations/1/_kernel/.ATTRIBUTES/VARIABLE_VALUE
         (bias path also changed from layer_with_weights-0 to _functional/_operations/1)

    ConvModule Dense (pointwise1/2), FFN Sequential layers, BatchNorm, LayerNorm,
    normLayer, channel_embed, and depthwise conv variables load correctly (same keys).
    """
    reader = tf.train.load_checkpoint(ckpt_path)
    ckpt_keys = set(reader.get_variable_to_shape_map().keys())
    n_fixed = 0

    def _assign(var, old_key):
        """Read tensor from old checkpoint and assign to var. Returns True on success."""
        nonlocal n_fixed
        suffix = '/.ATTRIBUTES/VARIABLE_VALUE'
        full_key = old_key + suffix
        if full_key not in ckpt_keys:
            return False
        tensor = tf.cast(reader.get_tensor(full_key), var.dtype)
        var.assign(tensor)
        n_fixed += 1
        return True

    def _get_kernel(layer):
        """Get the kernel variable regardless of whether it's named kernel or _kernel."""
        v = getattr(layer, '_kernel', None)
        if v is not None:
            return v
        return getattr(layer, 'kernel', None)

    # ── 1. MHA EinsumDense kernels ───────────────────────────────────────────
    MHA_DENSE_ATTRS = ['_query_dense', '_key_dense', '_value_dense', '_output_dense']

    def _fix_mha(mha_layer, ckpt_prefix):
        for dattr in MHA_DENSE_ATTRS:
            dense = getattr(mha_layer, dattr, None)
            if dense is None:
                continue
            kernel_var = _get_kernel(dense)
            if kernel_var is not None:
                _assign(kernel_var, f'{ckpt_prefix}/{dattr}/kernel')
            # bias loads fine (same path), no fix needed

    # Spatial attention MHA
    if hasattr(model, 'spatial_attn') and hasattr(model.spatial_attn, 'mha'):
        _fix_mha(model.spatial_attn.mha, 'net/spatial_attn/mha')

    # Encoder layer MHAs
    if hasattr(model, 'enc_layers'):
        for i, enc_layer in enumerate(model.enc_layers):
            if hasattr(enc_layer, 'mha'):
                _fix_mha(enc_layer.mha, f'net/enc_layers/{i}/mha')

    # ── 2. Direct Dense layers on ConformerEncoder ───────────────────────────
    # input_proj: projects stacked input features → d_model
    if hasattr(model, 'input_proj'):
        k = _get_kernel(model.input_proj)
        if k is not None:
            _assign(k, 'net/input_proj/kernel')

    # output dense (nClasses)
    if hasattr(model, 'dense'):
        k = _get_kernel(model.dense)
        if k is not None:
            _assign(k, 'net/dense/kernel')

    # ── 3. SpatialAttention Dense layers ────────────────────────────────────
    if hasattr(model, 'spatial_attn'):
        sa = model.spatial_attn
        for attr in ('proj', 'out_proj'):
            layer = getattr(sa, attr, None)
            if layer is not None:
                k = _get_kernel(layer)
                if k is not None:
                    _assign(k, f'net/spatial_attn/{attr}/kernel')

    # ── 4. Sequential inputLayer Dense sublayers ────────────────────────────
    # Each inputLayer is a Sequential: [InputLayer, Dense, Dropout]
    # Old path: inputLayer_N/layer_with_weights-0/{kernel,bias}
    # New path: inputLayer_N/_functional/_operations/1/{_kernel,bias}
    for idx, inp_model in enumerate(input_layers):
        # Find the Dense sublayer (skip InputLayer/Dropout)
        dense_layer = None
        for sublayer in inp_model.layers:
            if isinstance(sublayer, tf.keras.layers.Dense):
                dense_layer = sublayer
                break
        if dense_layer is None:
            continue
        prefix = f'inputLayer_{idx}/layer_with_weights-0'
        k = _get_kernel(dense_layer)
        if k is not None:
            _assign(k, f'{prefix}/kernel')
        if hasattr(dense_layer, 'bias') and dense_layer.bias is not None:
            _assign(dense_layer.bias, f'{prefix}/bias')

    log.info("  Checkpoint compat fix: assigned %d variables", n_fixed)


def run_inference(ckpt_dir=CKPT_DIR, data_dir=DATA_DIR):
    """Load the Conformer checkpoint, run full test-set inference.

    Returns dict with keys:
      logits          – (N, T_max, 41) float32, raw (pre-softmax) logits
      logitLengths    – (N,) int, valid timesteps per sample
      transcriptions  – list[str], ground-truth sentences
      greedy_per      – float, greedy-CTC phoneme error rate
    """
    log.info("Loading checkpoint from %s", ckpt_dir)
    args = OmegaConf.load(os.path.join(ckpt_dir, 'args.yaml'))
    args['loadDir']          = ckpt_dir
    args['outputDir']        = ckpt_dir
    args['mode']             = 'infer'
    args['loadCheckpointIdx'] = None

    if args.get('mixedPrecision', False):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # Use all 24 sessions at equal weight
    for x in range(len(args['dataset']['datasetProbabilityVal'])):
        args['dataset']['datasetProbabilityVal'][x] = 1.0
        args['dataset']['dataDir'][x] = data_dir

    args['testDir'] = 'test'

    tf.compat.v1.reset_default_graph()
    nsd = NeuralSequenceDecoder(args)

    # Fix all TF 2.15 → 2.21 checkpoint compatibility issues
    ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
    _fix_all_checkpoint_compat(nsd.model, nsd.inputLayers, ckpt_path)

    out = nsd.inference()

    # Decode ground-truth transcriptions from ASCII arrays
    transcriptions = _decode_transcriptions(out['transcriptions'])

    return {
        'logits':        out['logits'],          # (N, T_max, 41)
        'logitLengths':  out['logitLengths'],    # (N,)
        'transcriptions': transcriptions,
        'greedy_per':    float(out['cer']),       # field named 'cer' but it IS PER for speech
    }


def _decode_transcriptions(trans_array):
    """Convert (N, L) ASCII integer array to list of strings."""
    sentences = []
    for row in trans_array:
        end = np.argwhere(row == 0)
        if len(end) == 0:
            sentences.append(''.join(chr(c) for c in row if c > 0))
        else:
            sentences.append(''.join(chr(c) for c in row[:end[0, 0]]))
    return sentences


# ═══════════════════════════════════════════════════════════════════════════
# 2.  CTC PREFIX BEAM SEARCH  (pure Python, no Kaldi)
# ═══════════════════════════════════════════════════════════════════════════

def log_softmax(x):
    """Numerically stable log-softmax along last axis."""
    x = x - np.max(x, axis=-1, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))


def ctc_prefix_beam_search(logits, beam_size=50):
    """CTC prefix beam search on a single utterance.

    Args:
        logits: (T, V) raw logits; V=41, index 40=blank, 39=SIL, 0-38=phonemes
        beam_size: number of beams to keep

    Returns:
        list of (phoneme_index_tuple, log_score) sorted best-first
        (SIL tokens included in the returned sequences)
    """
    T, V = logits.shape
    log_probs = log_softmax(logits)   # (T, V)

    # Beam: dict of prefix_tuple → (log_p_blank, log_p_no_blank)
    NEG_INF = float('-inf')
    beams = {(): (0.0, NEG_INF)}      # empty prefix starts with p_blank=1

    for t in range(T):
        lp = log_probs[t]              # (V,)
        new_beams = defaultdict(lambda: (NEG_INF, NEG_INF))

        for prefix, (pb, pnb) in beams.items():
            # log total probability for this prefix
            p_total = np.logaddexp(pb, pnb)

            # ── Extend with blank ──────────────────────────────────────────
            new_pb = np.logaddexp(new_beams[prefix][0], p_total + lp[BLANK_IDX])
            new_beams[prefix] = (new_pb, new_beams[prefix][1])

            # ── Extend with each non-blank symbol ─────────────────────────
            for c in range(V - 1):        # 0..39 (phonemes + SIL, no blank)
                new_prefix = prefix + (c,)

                if len(prefix) > 0 and prefix[-1] == c:
                    # If same symbol as last: only p_blank contributes to p_nb
                    add_pnb = pb + lp[c]
                else:
                    add_pnb = p_total + lp[c]

                old_pb2, old_pnb2 = new_beams[new_prefix]
                new_beams[new_prefix] = (old_pb2, np.logaddexp(old_pnb2, add_pnb))

        # ── Prune to top beam_size ─────────────────────────────────────────
        def beam_score(item):
            pb, pnb = item[1]
            return np.logaddexp(pb, pnb)

        beams = dict(sorted(new_beams.items(), key=beam_score, reverse=True)[:beam_size])

    # ── Final ranking ──────────────────────────────────────────────────────
    results = []
    for prefix, (pb, pnb) in beams.items():
        score = np.logaddexp(pb, pnb)
        results.append((prefix, score))
    results.sort(key=lambda x: -x[1])
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 3.  PHONEME → WORD  CONVERSION  (CMU dict + DP)
# ═══════════════════════════════════════════════════════════════════════════

def build_phoneme_trie():
    """Build a trie mapping stripped-stress phoneme tuples → list of words.

    Returns dict of dict: trie[phone1][phone2]...[phoneN]['#WORDS'] = [word, ...]
    We use a flat dict: (phone_tuple) → [word, ...]  for simplicity / speed.
    """
    import cmudict as cmudict_lib
    log.info("Building CMU pronunciation reverse-lookup table …")

    phone_to_words = defaultdict(list)
    for word, phones in cmudict_lib.entries():
        # Skip words starting with non-alphabetic (e.g. "'bout", numbers)
        if not word[0].isalpha():
            continue
        # Strip stress markers: 'AH0' → 'AH', 'EY1' → 'EY'
        stripped = tuple(p.rstrip('012') for p in phones)
        phone_to_words[stripped].append(word)

    log.info("  Reverse lookup: %d unique pronunciation sequences", len(phone_to_words))
    return dict(phone_to_words)


def phones_to_words_dp(phone_seq, phone_to_words, beam_size=20):
    """DP segmentation: phoneme index sequence → best word-string hypotheses.

    Args:
        phone_seq: tuple of phoneme indices (0-38 = phonemes, 39 = SIL ignored)
        phone_to_words: dict from build_phoneme_trie()
        beam_size: max word-segmentation hypotheses to keep

    Returns:
        list of word strings (space-separated sentences), up to beam_size
    """
    # Convert indices to phoneme label strings, drop SIL
    phone_labels = tuple(
        IDX_TO_PHONE[idx] for idx in phone_seq
        if idx < SIL_IDX   # keep only real phonemes, skip SIL
    )

    N = len(phone_labels)
    if N == 0:
        return ['']

    # dp[i] = list of (word_list, score) ending at position i
    # score = number of words (shorter segmentations preferred)
    # We keep beam_size hypotheses per position.
    dp = [[] for _ in range(N + 1)]
    dp[0] = [([], 0)]       # empty word list at start

    for start in range(N):
        if not dp[start]:
            continue
        # Try all end positions
        for end in range(start + 1, N + 1):
            segment = phone_labels[start:end]
            if segment in phone_to_words:
                words_here = phone_to_words[segment]
                for hyp_words, hyp_score in dp[start]:
                    for w in words_here:
                        new_entry = (hyp_words + [w], hyp_score + 1)
                        dp[end].append(new_entry)
                # Prune dp[end] to beam_size (prefer fewer words = more coverage)
                if len(dp[end]) > beam_size * 5:
                    dp[end].sort(key=lambda x: x[1])
                    dp[end] = dp[end][:beam_size * 5]

    if not dp[N]:
        # No full segmentation found — fall back to subword coverage
        # Return the best partial coverage we have (longest covered prefix)
        best_partial = _best_partial_coverage(phone_labels, phone_to_words)
        return [best_partial] if best_partial else ['']

    # Deduplicate and return top hypotheses
    seen = set()
    results = []
    for word_list, _ in sorted(dp[N], key=lambda x: x[1]):
        sentence = ' '.join(word_list)
        if sentence not in seen:
            seen.add(sentence)
            results.append(sentence)
        if len(results) >= beam_size:
            break

    return results


def _best_partial_coverage(phone_labels, phone_to_words):
    """Greedy left-to-right word matching for phoneme sequences with no full parse."""
    words = []
    pos = 0
    N = len(phone_labels)
    while pos < N:
        best_end = -1
        best_word = ''
        # Try longest match first
        for end in range(min(pos + 12, N), pos, -1):   # max word length ≈ 12 phones
            segment = phone_labels[pos:end]
            if segment in phone_to_words:
                best_end = end
                best_word = phone_to_words[segment][0]
                break
        if best_end == -1:
            pos += 1          # skip unrecognised phoneme
        else:
            words.append(best_word)
            pos = best_end
    return ' '.join(words)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  LANGUAGE MODEL  LOADING & SCORING
# ═══════════════════════════════════════════════════════════════════════════

def load_lm(model_id, cache_dir=None, hf_token=None):
    """Load an autoregressive LM via HuggingFace transformers (PyTorch).

    Returns (model, tokenizer) on CUDA if available.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading LM: %s", model_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info("  Device: %s", device)

    tok_kwargs = {'cache_dir': cache_dir}
    if hf_token:
        tok_kwargs['token'] = hf_token

    kwargs = {'cache_dir': cache_dir, 'low_cpu_mem_usage': True}
    if hf_token:
        kwargs['token'] = hf_token
    if 'gemma' in model_id.lower():
        kwargs['torch_dtype'] = torch.bfloat16
        kwargs['trust_remote_code'] = True
    else:
        kwargs['torch_dtype'] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model = model.to(device).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("  Loaded %s  (%.0fM params)", model_id,
             sum(p.numel() for p in model.parameters()) / 1e6)
    return model, tokenizer


def score_lm_batch(model, tokenizer, texts, device=None):
    """Return per-sentence log P(text) under the LM.

    Args:
        texts: list of strings
    Returns:
        np.ndarray of shape (len(texts),) with log-probabilities
    """
    import torch

    if device is None:
        device = next(model.parameters()).device

    if not texts:
        return np.array([])

    # Tokenise with padding
    enc = tokenizer(texts, return_tensors='pt', padding=True,
                    truncation=True, max_length=128).to(device)

    with torch.no_grad():
        out = model(**enc)
        # out.logits: (B, T, V)
        lp = torch.nn.functional.log_softmax(out.logits.float(), dim=-1)

    lp = lp.cpu().numpy()
    input_ids = enc['input_ids'].cpu().numpy()
    attn_mask  = enc['attention_mask'].cpu().numpy()

    scores = np.zeros(len(texts))
    for i in range(len(texts)):
        n_tok = int(attn_mask[i].sum())
        s = 0.0
        for j in range(1, n_tok):
            s += lp[i, j - 1, input_ids[i, j]]
        scores[i] = s

    return scores


# ═══════════════════════════════════════════════════════════════════════════
# 5.  FULL DECODE LOOP
# ═══════════════════════════════════════════════════════════════════════════

def decode_with_lm(logits, logit_lengths, lm_model, lm_tokenizer,
                   phone_to_words,
                   beam_size=50, lm_nbest=20,
                   acoustic_scale=0.5, lm_weight=0.5,
                   lm_name='lm'):
    """Decode all utterances with CTC beam search + LM rescoring.

    Returns list of decoded sentence strings.
    """
    import torch
    device = next(lm_model.parameters()).device

    N = len(logit_lengths)
    decoded_sentences = []
    decode_times = []

    for i in range(N):
        t0 = time.time()
        T = int(logit_lengths[i])
        raw_logits = logits[i, :T, :]    # (T, 41)

        # ── CTC beam search ───────────────────────────────────────────────
        nbest = ctc_prefix_beam_search(raw_logits, beam_size=beam_size)

        # ── Convert each N-best phoneme sequence to word hypotheses ───────
        word_hyps = {}    # sentence_text → best_combined_score
        for phone_seq, ac_score in nbest[:lm_nbest]:
            word_seqs = phones_to_words_dp(phone_seq, phone_to_words,
                                           beam_size=10)
            for sentence in word_seqs:
                if sentence not in word_hyps:
                    word_hyps[sentence] = (ac_score, None)   # lm_score filled below
                else:
                    # Keep higher acoustic score for same sentence
                    if ac_score > word_hyps[sentence][0]:
                        word_hyps[sentence] = (ac_score, word_hyps[sentence][1])

        if not word_hyps:
            decoded_sentences.append('')
            decode_times.append(time.time() - t0)
            continue

        # ── Batch LM scoring ──────────────────────────────────────────────
        hyp_texts = list(word_hyps.keys())
        lm_scores = score_lm_batch(lm_model, lm_tokenizer, hyp_texts, device=device)

        # ── Combine scores and select best ────────────────────────────────
        best_sent = ''
        best_score = float('-inf')
        for j, sentence in enumerate(hyp_texts):
            ac_score = word_hyps[sentence][0]
            combined = acoustic_scale * ac_score + lm_weight * lm_scores[j]
            if combined > best_score:
                best_score = combined
                best_sent = sentence

        decoded_sentences.append(best_sent)
        decode_times.append(time.time() - t0)

        if (i + 1) % 50 == 0 or i == 0:
            log.info("  [%s] %d/%d done  (avg %.2fs/utt)",
                     lm_name, i + 1, N, np.mean(decode_times))

    return decoded_sentences, decode_times


def decode_greedy_phonemes(logits, logit_lengths):
    """CTC greedy decode to get PER baseline (phoneme sequences only).

    Returns list of decoded phoneme index tuples (collapse repeated, remove blank/SIL).
    """
    results = []
    for i in range(len(logit_lengths)):
        T = int(logit_lengths[i])
        raw = logits[i, :T, :]
        pred_ids = np.argmax(raw, axis=-1)   # (T,)
        # CTC collapse: remove consecutive duplicates, then remove blank
        prev = -1
        seq = []
        for idx in pred_ids:
            if idx != prev:
                seq.append(int(idx))
                prev = int(idx)
        seq = [x for x in seq if x != BLANK_IDX]
        results.append(tuple(seq))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 6.  METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(decoded, ground_truth, logit_lengths):
    """Compute CER, WER, WPM.

    Args:
        decoded:       list[str] decoded word-level sentences
        ground_truth:  list[str] reference sentences
        logit_lengths: (N,) int, output logit lengths (before LM)

    Returns dict with cer, wer, wpm (all floats)
    """
    total_char_err = 0; total_chars = 0
    total_word_err = 0; total_words = 0

    for dec, ref in zip(decoded, ground_truth):
        dec_chars = list(dec.lower())
        ref_chars = list(ref.lower())
        total_char_err += edit_distance(ref_chars, dec_chars)
        total_chars    += max(len(ref_chars), 1)

        dec_words = dec.lower().split()
        ref_words = ref.lower().split()
        total_word_err += edit_distance(ref_words, dec_words)
        total_words    += max(len(ref_words), 1)

    cer = total_char_err / total_chars
    wer = total_word_err / total_words

    # WPM: total decoded words / total audio duration in minutes
    total_audio_sec = float(np.sum(logit_lengths)) * FRAME_DURATION_SEC
    total_audio_min = total_audio_sec / 60.0
    wpm = (total_words / total_audio_min) if total_audio_min > 0 else 0.0

    return dict(cer=cer, wer=wer, wpm=wpm)


def compute_per(greedy_phone_seqs, ground_truth_texts, phone_to_words):
    """Approximate PER by comparing greedy-decoded phonemes to reference phonemes.

    Reference phonemes are obtained by looking up ground-truth words in CMU dict
    and taking the first pronunciation.
    """
    import cmudict as cmudict_lib
    word2phones = {}
    for word, phones in cmudict_lib.entries():
        if word not in word2phones and word[0].isalpha():
            word2phones[word] = tuple(p.rstrip('012') for p in phones)

    total_phone_err = 0
    total_phones    = 0

    for hyp_seq, ref_text in zip(greedy_phone_seqs, ground_truth_texts):
        # Build reference phoneme sequence from words
        ref_phones = []
        for word in ref_text.lower().split():
            if word in word2phones:
                ref_phones.extend(word2phones[word])

        if not ref_phones:
            continue

        # Convert hyp phoneme indices to labels (skip SIL)
        hyp_phones = [IDX_TO_PHONE[idx] for idx in hyp_seq if idx < SIL_IDX]

        total_phone_err += edit_distance(ref_phones, hyp_phones)
        total_phones    += len(ref_phones)

    return total_phone_err / total_phones if total_phones > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='LM Pipeline Evaluation')
    parser.add_argument('--lm', choices=['gpt2', 'gemma', 'both'], default='both',
                        help='Which LM(s) to evaluate (default: both)')
    parser.add_argument('--beam-size', type=int, default=50,
                        help='CTC beam size (default: 50)')
    parser.add_argument('--lm-nbest', type=int, default=30,
                        help='N-best phoneme sequences to convert to words (default: 30)')
    parser.add_argument('--acoustic-scale', type=float, default=0.5,
                        help='Acoustic model score weight (default: 0.5)')
    parser.add_argument('--lm-weight', type=float, default=0.5,
                        help='LM score weight alpha (default: 0.5)')
    parser.add_argument('--cache-dir', type=str, default='/root/.cache/huggingface',
                        help='HuggingFace cache directory')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--hf-token', type=str,
                        default=os.environ.get('HF_TOKEN', ''),
                        help='HuggingFace token for gated models (or set HF_TOKEN env var)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    log.info("=" * 65)
    log.info("  LM Pipeline Evaluation")
    log.info("  Decoder  : %s", CKPT_DIR)
    log.info("  LM(s)    : %s", args.lm)
    log.info("  Beam     : %d  |  N-best words: %d", args.beam_size, args.lm_nbest)
    log.info("  α_ac=%.2f  α_lm=%.2f", args.acoustic_scale, args.lm_weight)
    log.info("=" * 65)

    # ── Step 1: Inference ──────────────────────────────────────────────────
    log.info("\n[1/5]  Running Conformer inference …")
    inf = run_inference(ckpt_dir=CKPT_DIR, data_dir=DATA_DIR)
    logits       = inf['logits']
    logit_lengths = inf['logitLengths']
    ground_truth  = inf['transcriptions']
    greedy_per_internal = inf['greedy_per']

    log.info("  Samples  : %d", len(logit_lengths))
    log.info("  Greedy PER (internal): %.4f", greedy_per_internal)

    # ── Step 2: Build phoneme → words lookup ──────────────────────────────
    log.info("\n[2/5]  Building CMU pronunciation reverse lookup …")
    phone_to_words = build_phoneme_trie()

    # ── Step 3: Greedy PER from our beam search (sanity check) ────────────
    log.info("\n[3/5]  CTC greedy decode (PER baseline) …")
    greedy_seqs = decode_greedy_phonemes(logits, logit_lengths)
    per_ours = compute_per(greedy_seqs, ground_truth, phone_to_words)
    log.info("  PER (greedy, word-aligned): %.4f", per_ours)

    results = {
        'config': {
            'decoder_ckpt': CKPT_DIR,
            'beam_size': args.beam_size,
            'lm_nbest': args.lm_nbest,
            'acoustic_scale': args.acoustic_scale,
            'lm_weight': args.lm_weight,
        },
        'neural_decoder_per': round(greedy_per_internal, 6),
        'lm_results': {},
    }

    lms_to_run = []
    if args.lm in ('gpt2', 'both'):
        lms_to_run.append(('gpt2', GPT2_MODEL_ID))
    if args.lm in ('gemma', 'both'):
        lms_to_run.append(('gemma3_270m', GEMMA_MODEL_ID))

    # ── Step 4: LM decoding for each model ────────────────────────────────
    for lm_tag, lm_id in lms_to_run:
        log.info("\n[4/5]  LM decoding with %s …", lm_id)

        # Free TF GPU memory before loading PyTorch model
        tf.keras.backend.clear_session()

        lm_model, lm_tokenizer = load_lm(lm_id, cache_dir=args.cache_dir,
                                          hf_token=args.hf_token or None)

        t_decode_start = time.time()
        decoded_sents, decode_times = decode_with_lm(
            logits, logit_lengths,
            lm_model, lm_tokenizer,
            phone_to_words,
            beam_size=args.beam_size,
            lm_nbest=args.lm_nbest,
            acoustic_scale=args.acoustic_scale,
            lm_weight=args.lm_weight,
            lm_name=lm_tag,
        )
        t_decode_total = time.time() - t_decode_start

        # ── Step 5: Metrics ───────────────────────────────────────────────
        log.info("\n[5/5]  Computing metrics for %s …", lm_tag)
        metrics = compute_metrics(decoded_sents, ground_truth, logit_lengths)

        # WPM note: use ground-truth word count / total audio duration
        total_gt_words = sum(len(s.split()) for s in ground_truth)
        total_audio_min = float(np.sum(logit_lengths)) * FRAME_DURATION_SEC / 60.0
        wpm_gt  = total_gt_words / total_audio_min

        # Also compute decode-time WPM (how many words decoded per minute of compute)
        n_decoded_words = sum(len(s.split()) for s in decoded_sents)
        wpm_decoded = n_decoded_words / (t_decode_total / 60.0)

        log.info("")
        log.info("  ┌─────────────────────────────────┐")
        log.info("  │  Results: %-22s │", lm_tag)
        log.info("  ├─────────────────────────────────┤")
        log.info("  │  PER  (neural decoder) = %.4f  │", greedy_per_internal)
        log.info("  │  CER  (LM decoded)     = %.4f  │", metrics['cer'])
        log.info("  │  WER  (LM decoded)     = %.4f  │", metrics['wer'])
        log.info("  │  WPM  (audio-based)    = %.1f  │", wpm_gt)
        log.info("  │  WPM  (decode-speed)   = %.1f  │", wpm_decoded)
        log.info("  └─────────────────────────────────┘")

        results['lm_results'][lm_tag] = {
            'lm_model_id': lm_id,
            'per':  round(greedy_per_internal, 6),
            'cer':  round(metrics['cer'], 6),
            'wer':  round(metrics['wer'], 6),
            'wpm_audio':  round(wpm_gt, 2),
            'wpm_decode': round(wpm_decoded, 2),
            'n_samples': len(decoded_sents),
            'avg_decode_time_sec': round(float(np.mean(decode_times)), 4),
            'total_decode_time_sec': round(t_decode_total, 2),
        }

        # Save a sample of decoded vs ground-truth for inspection
        sample_comparisons = []
        for i in range(min(20, len(decoded_sents))):
            sample_comparisons.append({
                'ref':  ground_truth[i],
                'hyp':  decoded_sents[i],
            })
        results['lm_results'][lm_tag]['sample_comparisons'] = sample_comparisons

        # Clean up LM from memory
        import torch
        del lm_model, lm_tokenizer
        torch.cuda.empty_cache()

    # ── Save results ──────────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir, 'lm_pipeline_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    log.info("\nResults saved → %s", out_path)
    log.info("\n" + "=" * 65)
    log.info("  FINAL SUMMARY")
    log.info("=" * 65)
    log.info("  Neural decoder PER : %.4f", greedy_per_internal)
    for lm_tag, r in results['lm_results'].items():
        log.info("  %s:", lm_tag)
        log.info("    PER=%.4f  CER=%.4f  WER=%.4f  WPM(audio)=%.1f",
                 r['per'], r['cer'], r['wer'], r['wpm_audio'])


if __name__ == '__main__':
    main()
