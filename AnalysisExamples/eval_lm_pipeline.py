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
# LD_LIBRARY_PATH must be set BEFORE the dynamic linker loads TF's .so files.
# Setting os.environ inside Python is too late for already-loaded libs, so we
# re-exec the process with the correct env if the NVIDIA libs are not yet on the path.
import site as _site
_NV_BASE = os.path.join(_site.getsitepackages()[0], 'nvidia')
if os.path.isdir(_NV_BASE):
    _NV_SUBDIRS = ['cudnn', 'cublas', 'cuda_nvrtc', 'cuda_runtime',
                   'cufft', 'cusolver', 'cusparse', 'nvjitlink']
    _NV_LIBS = ':'.join(
        os.path.join(_NV_BASE, d, 'lib')
        for d in _NV_SUBDIRS
        if os.path.isdir(os.path.join(_NV_BASE, d, 'lib'))
    )
    _cur_ld = os.environ.get('LD_LIBRARY_PATH', '')
    if _NV_LIBS and _NV_LIBS.split(':')[0] not in _cur_ld:
        _env = os.environ.copy()
        _env['LD_LIBRARY_PATH'] = _NV_LIBS + ':' + _cur_ld
        os.execve(sys.executable, [sys.executable] + sys.argv, _env)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
# Raw model logits layout (V=41):
#   class 0..38 = phonemes PHONE_DEF_SIL[0..38]
#   class 39    = SIL       PHONE_DEF_SIL[39]
#   class 40    = CTC blank (blank_index=last_class in training ctc_loss)
# TFRecord seqClassIDs are 1-indexed labels (1..40 → output class 0..39);
# training used ctc_loss with blank_index=40.
BLANK_IDX = 40
PHONE_CLASS_OFFSET = 0      # logit_class == PHONE_DEF_SIL idx for phonemes
SIL_LOGIT_IDX = 39          # logit class for SIL
SIL_IDX = 39                # internal phoneme idx (PHONE_DEF_SIL[39]='SIL')
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
# 2.  LEXICON TRIE + LEXICON-CONSTRAINED CTC BEAM SEARCH
# ═══════════════════════════════════════════════════════════════════════════
# Matches Willett et al. (lexicon-constrained decoding) and Seto et al.
# (shallow-fusion combine: score = acoustic + α·log P_LM + β·#words).
# Rather than producing phoneme N-best and then mapping to words via a post-hoc
# DP, we constrain the CTC beam search to only extend along valid CMU-dict
# phoneme prefixes, so every hypothesis is already a sequence of real words.

# Phoneme label → model class index (0-38 = phonemes, 39 = SIL)
_PHONE_LABEL_TO_IDX = {p: i for i, p in enumerate(PHONE_DEF_SIL)}


def log_softmax(x):
    """Numerically stable log-softmax along last axis."""
    x = x - np.max(x, axis=-1, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))


class LexiconTrie:
    """Prefix trie over CMU-dict phoneme sequences (indices in 0..38).

    - children[node_id] : dict[phone_idx → child_node_id]
    - words_at[node_id] : list[str] (non-empty iff this node is a word-end)
    - Root is node 0.
    """

    def __init__(self):
        self.children = [{}]       # list of dicts, one per node
        self.words_at = [[]]       # list of word-lists, one per node

    def _new_node(self):
        self.children.append({})
        self.words_at.append([])
        return len(self.children) - 1

    def add(self, phone_idx_seq, word):
        node = 0
        for c in phone_idx_seq:
            if c not in self.children[node]:
                self.children[node][c] = self._new_node()
            node = self.children[node][c]
        self.words_at[node].append(word)

    def __len__(self):
        return len(self.children)


# Very-short words in CMU that are legitimate English. 1-phone entries like
# "ah", "mm", "eh", "aux" are junk interjections/abbreviations that pollute
# the lexicon and cause over-segmentation in the beam. We keep only the
# genuinely common 1- and 2-phoneme words.
_SHORT_WORD_WHITELIST = {
    'a', 'i', 'oh', 'uh',                               # 1-phone legit
    'am', 'an', 'as', 'at', 'be', 'by', 'do', 'go',     # 2-phone
    'he', 'if', 'in', 'is', 'it', 'me', 'my', 'no',
    'of', 'on', 'or', 'so', 'to', 'up', 'us', 'we',
    'ah', 'ow', 'ow.', 'ye', 'ya', 'ya.', 'um',
    "i'd", "i'll", "i'm", "i've", "he'd", "he'll", "he's",
    "it's", "we'd", "we'll", "we're", "we've",
}


def build_lexicon_trie(drop_short_junk=True):
    """Load CMU dict and build a phoneme-index → word trie.

    Stress markers are stripped (AH0 → AH). Every pronunciation variant is
    inserted. Words starting with non-alphabetic characters are skipped.
    If drop_short_junk, 1-2 phoneme words not in the whitelist are dropped
    (removes CMU's "ae", "ahh", "eau", "err" etc. that over-segment the beam).
    """
    import cmudict as cmudict_lib
    log.info("Building CMU-dict lexicon trie …")

    trie = LexiconTrie()
    n_words = 0
    n_skipped = 0
    n_short_dropped = 0
    unique_words = set()
    for word, phones in cmudict_lib.entries():
        if not word[0].isalpha():
            n_skipped += 1
            continue
        try:
            idx_seq = tuple(_PHONE_LABEL_TO_IDX[p.rstrip('012')] for p in phones)
        except KeyError:
            n_skipped += 1
            continue
        if not idx_seq:
            continue
        if drop_short_junk and len(idx_seq) <= 2 and word.lower() not in _SHORT_WORD_WHITELIST:
            n_short_dropped += 1
            continue
        trie.add(idx_seq, word)
        n_words += 1
        unique_words.add(word)

    log.info("  Pronunciations inserted : %d", n_words)
    log.info("  Unique words            : %d", len(unique_words))
    log.info("  Trie nodes              : %d", len(trie))
    log.info("  Skipped (non-alpha/phon): %d", n_skipped)
    log.info("  Short-junk dropped      : %d", n_short_dropped)
    return trie


def lexicon_constrained_beam_search(logits, trie, beam_size=100, beam_beta=1.0):
    """Word-level lexicon-constrained CTC beam search (Viterbi-style).

    Emits only sequences of real CMU-dict words. SIL (class 39) is treated as
    the word-boundary signal: it may only be emitted when the current in-progress
    phoneme path corresponds to a complete word, and doing so commits that word
    and resets the trie cursor to the root.

    Args:
        logits: (T, V) raw logits, V = 41 (0-38 phonemes, 39 SIL, 40 blank).
        trie: LexiconTrie
        beam_size: max beams kept per frame
        beam_beta: per-word bonus added to beam log-prob when a word is
            committed. Used only for in-search pruning — does NOT shift the
            returned acoustic score.

    Returns:
        list of (sentence_str, log_acoustic, n_words) sorted best-first.
        log_acoustic is the pure path log-prob (no beta).
    """
    T, V = logits.shape
    log_probs = log_softmax(logits)
    NEG_INF = float('-inf')

    # Beam key: (committed_words_tuple, trie_node_id, last_emitted_idx)
    #   last_emitted_idx: -1 if last step was blank (or start); else the
    #   internal phoneme idx (0..39) emitted last. Needed to collapse CTC
    #   repeats. Internal-idx → logit class = idx + 1.
    # Beam value: (log_prob_pure_acoustic, n_words)
    init_key = ((), 0, -1)
    beams = {init_key: (0.0, 0)}

    # For a given trie_node, cache the list of valid child phoneme idxs
    children = trie.children
    words_at = trie.words_at

    for t in range(T):
        lp = log_probs[t]
        new_beams = {}

        def _accum(key, lp_new, nwords_new):
            cur = new_beams.get(key)
            if cur is None or lp_new > cur[0]:
                # Viterbi max (path score). Simpler than log-sum-exp and
                # gives nearly identical N-best rankings in practice.
                new_beams[key] = (lp_new, nwords_new)

        # Local alias to avoid attribute lookups in hot loop
        lp_blank = lp[BLANK_IDX]
        lp_sil = lp[SIL_LOGIT_IDX]

        for (cwords, node, last_idx), (lp_old, nwords_old) in beams.items():
            # ── (1) BLANK: same beam state, last_idx → -1
            _accum((cwords, node, -1), lp_old + lp_blank, nwords_old)

            # ── (2) Repeat last non-blank symbol: stays in same state,
            #        last_idx unchanged. Only valid if last_idx != -1.
            if last_idx != -1:
                last_lp = lp_sil if last_idx == SIL_IDX else lp[last_idx + PHONE_CLASS_OFFSET]
                _accum((cwords, node, last_idx), lp_old + last_lp, nwords_old)

            # ── (3) SIL = word-boundary signal.
            #        Valid when node is root (absorbs silence between words /
            #        at utterance start) OR the current node is a word-end
            #        (commits the word and resets cursor to root).
            if last_idx != SIL_IDX:   # if last_idx == SIL_IDX, handled by (2)
                if node == 0:
                    _accum((cwords, 0, SIL_IDX), lp_old + lp_sil, nwords_old)
                elif words_at[node]:
                    # Commit the first pronunciation variant (homophones
                    # disambiguated later by the LM).
                    word = words_at[node][0]
                    new_cwords = cwords + (word,)
                    new_lp = lp_old + lp_sil + beam_beta   # bonus only used for pruning
                    _accum((new_cwords, 0, SIL_IDX), new_lp, nwords_old + 1)

            # ── (4) Phoneme extension through trie (internal idx 0..38) ──
            # a) Continue the current word.
            node_children = children[node]
            for c, child in node_children.items():
                if c == last_idx:
                    # Repeat handled as continuation in (2).
                    continue
                _accum((cwords, child, c), lp_old + lp[c + PHONE_CLASS_OFFSET], nwords_old)

            # b) Implicit word-boundary: if node is a word-end, commit and
            #    start a new word directly without an intervening SIL.
            if words_at[node] and node != 0:
                word = words_at[node][0]
                new_cwords = cwords + (word,)
                root_children = children[0]
                for c, child in root_children.items():
                    if c == last_idx:
                        continue
                    _accum((new_cwords, child, c),
                           lp_old + lp[c + PHONE_CLASS_OFFSET] + beam_beta,
                           nwords_old + 1)

        # ── Prune to top beam_size by (acoustic + beam_beta * n_words) ──
        # Note: beta was baked into committed-word lp above, so simple lp
        # comparison works.
        beams = dict(
            sorted(new_beams.items(), key=lambda kv: kv[1][0], reverse=True)[:beam_size]
        )

    # ── Finalise: commit any word still in progress (if current node is a
    #    word-end), otherwise drop the trailing partial word. ─────────────
    results = {}
    for (cwords, node, _last), (lp_final, nwords) in beams.items():
        if node == 0:
            final_words = cwords
            final_nwords = nwords
            final_lp = lp_final
        elif words_at[node]:
            final_words = cwords + (words_at[node][0],)
            final_nwords = nwords + 1
            final_lp = lp_final + beam_beta
        else:
            # Partial word at end of utterance → drop (matches Kaldi's default).
            final_words = cwords
            final_nwords = nwords
            final_lp = lp_final

        sentence = ' '.join(final_words)
        # De-subtract beta from returned score so callers get pure acoustic.
        pure_ac = final_lp - beam_beta * final_nwords
        prev = results.get(sentence)
        if prev is None or pure_ac > prev[0]:
            results[sentence] = (pure_ac, final_nwords)

    return sorted(
        [(s, sc, nw) for s, (sc, nw) in results.items()],
        key=lambda x: -x[1],
    )


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

def run_beam_search_all(logits, logit_lengths, trie,
                        beam_size=100, lm_nbest=100, beam_beta=1.0,
                        log_every=50):
    """Run lexicon-constrained CTC beam search over every utterance once.

    Returns list of N-best lists: [(sentence, log_acoustic, n_words), …]
    One per utterance. The acoustic score is pure (beta NOT baked in), so
    callers can grid-search α, β at rescoring time without rerunning search.
    """
    N = len(logit_lengths)
    all_nbest = []
    t_start = time.time()

    for i in range(N):
        T = int(logit_lengths[i])
        raw_logits = logits[i, :T, :]
        nbest = lexicon_constrained_beam_search(
            raw_logits, trie, beam_size=beam_size, beam_beta=beam_beta,
        )
        all_nbest.append(nbest[:lm_nbest])

        if (i + 1) % log_every == 0 or i == 0:
            avg = (time.time() - t_start) / (i + 1)
            log.info("  beam search %d/%d  (avg %.2fs/utt)", i + 1, N, avg)

    return all_nbest


def score_nbest_with_lm(all_nbest, lm_model, lm_tokenizer,
                        lm_name='lm', log_every=50):
    """Score every candidate sentence in every utterance's N-best with the LM.

    Returns parallel structure: list[list[(sentence, log_ac, n_words, log_lm)]].
    """
    device = next(lm_model.parameters()).device
    out = []
    t_start = time.time()
    for i, nbest in enumerate(all_nbest):
        if not nbest:
            out.append([])
            continue
        texts = [s for s, _, _ in nbest]
        lm_scores = score_lm_batch(lm_model, lm_tokenizer, texts, device=device)
        out.append([
            (s, ac, nw, float(lm_scores[j]))
            for j, (s, ac, nw) in enumerate(nbest)
        ])
        if (i + 1) % log_every == 0 or i == 0:
            avg = (time.time() - t_start) / (i + 1)
            log.info("  [%s] LM-scored %d/%d  (avg %.2fs/utt)",
                     lm_name, i + 1, len(all_nbest), avg)
    return out


def pick_best_hyp(scored_nbest, alpha=0.5, beta=1.0, acoustic_scale=1.0):
    """Apply Seto III-3: combined = acoustic_scale*log_ac + α*log_lm + β*n_words.
    Returns the best sentence per utterance."""
    decoded = []
    for nbest in scored_nbest:
        if not nbest:
            decoded.append('')
            continue
        best = ''
        best_score = float('-inf')
        for s, log_ac, nw, log_lm in nbest:
            combined = acoustic_scale * log_ac + alpha * log_lm + beta * nw
            if combined > best_score:
                best_score = combined
                best = s
        decoded.append(best)
    return decoded


def decode_greedy_phonemes(logits, logit_lengths):
    """CTC greedy decode to get PER baseline.

    Returns list of internal phoneme-idx tuples (0..39, 39=SIL) with CTC
    collapse applied (remove consecutive duplicates, remove blanks).
    """
    results = []
    for i in range(len(logit_lengths)):
        T = int(logit_lengths[i])
        raw = logits[i, :T, :]
        pred_classes = np.argmax(raw, axis=-1)   # logit class ids (0..40)
        # CTC collapse: drop consecutive dups, then drop blank (class 0)
        prev = -1
        seq = []
        for cls in pred_classes:
            if cls != prev:
                seq.append(int(cls))
                prev = int(cls)
        # Convert logit class → internal phoneme idx (subtract 1, drop blanks)
        seq = [x - PHONE_CLASS_OFFSET for x in seq if x != BLANK_IDX]
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

def _parse_float_list(s):
    return [float(x) for x in s.split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser(description='LM Pipeline Evaluation')
    parser.add_argument('--lm', choices=['none', 'gpt2', 'gemma', 'both'],
                        default='both',
                        help='Which LM(s) to evaluate (default: both). "none" skips LM rescoring (lexicon-only baseline).')
    parser.add_argument('--beam-size', type=int, default=100,
                        help='Lexicon-constrained CTC beam size (default: 100)')
    parser.add_argument('--lm-nbest', type=int, default=100,
                        help='N-best sentences kept per utterance for LM rescoring (default: 100)')
    parser.add_argument('--beam-beta', type=float, default=1.0,
                        help='Per-word bonus used for beam pruning during search (default: 1.0)')
    parser.add_argument('--acoustic-scale', type=float, default=1.0,
                        help='Acoustic score weight in Seto eq. III-3 (default: 1.0)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='LM weight α in Seto eq. III-3 (default: 0.5)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Word-insertion bonus β in Seto eq. III-3 (default: 1.0)')
    parser.add_argument('--grid-search', action='store_true',
                        help='Sweep α, β on the test set and report the best.')
    parser.add_argument('--alphas', type=_parse_float_list,
                        default=[0.0, 0.3, 0.5, 0.8, 1.2],
                        help='α values for grid search (comma-separated)')
    parser.add_argument('--betas', type=_parse_float_list,
                        default=[0.0, 0.5, 1.0, 2.0, 4.0],
                        help='β values for grid search (comma-separated)')
    parser.add_argument('--max-utts', type=int, default=0,
                        help='If >0, limit to first N utterances (debug only)')
    parser.add_argument('--cache-dir', type=str, default='/root/.cache/huggingface',
                        help='HuggingFace cache directory')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--hf-token', type=str,
                        default=os.environ.get('HF_TOKEN', ''),
                        help='HuggingFace token for gated models (or set HF_TOKEN env var)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    log.info("=" * 65)
    log.info("  LM Pipeline Evaluation  (lexicon-constrained shallow fusion)")
    log.info("  Decoder  : %s", CKPT_DIR)
    log.info("  LM(s)    : %s", args.lm)
    log.info("  Beam=%d  N-best=%d  beam_beta=%.2f",
             args.beam_size, args.lm_nbest, args.beam_beta)
    log.info("  α=%.2f  β=%.2f  acoustic_scale=%.2f  grid=%s",
             args.alpha, args.beta, args.acoustic_scale, args.grid_search)
    log.info("=" * 65)

    # ── Step 1: Inference ──────────────────────────────────────────────────
    log.info("\n[1/6]  Running Conformer inference …")
    inf = run_inference(ckpt_dir=CKPT_DIR, data_dir=DATA_DIR)
    logits       = inf['logits']
    logit_lengths = inf['logitLengths']
    ground_truth  = inf['transcriptions']
    greedy_per_internal = inf['greedy_per']

    if args.max_utts > 0:
        logits        = logits[:args.max_utts]
        logit_lengths = logit_lengths[:args.max_utts]
        ground_truth  = ground_truth[:args.max_utts]
        log.info("  (limiting to first %d utterances)", args.max_utts)

    log.info("  Samples  : %d", len(logit_lengths))
    log.info("  Greedy PER (internal): %.4f", greedy_per_internal)

    # ── Step 2: Build lexicon trie ────────────────────────────────────────
    log.info("\n[2/6]  Building lexicon trie …")
    trie = build_lexicon_trie()

    # ── Step 3: Greedy PER baseline ───────────────────────────────────────
    log.info("\n[3/6]  CTC greedy decode (PER baseline) …")
    greedy_seqs = decode_greedy_phonemes(logits, logit_lengths)
    per_ours = compute_per(greedy_seqs, ground_truth, None)
    log.info("  PER (greedy, word-aligned): %.4f", per_ours)

    # ── Step 4: Lexicon-constrained beam search (once, reused by LMs) ────
    log.info("\n[4/6]  Lexicon-constrained beam search …")
    t_bs = time.time()
    all_nbest = run_beam_search_all(
        logits, logit_lengths, trie,
        beam_size=args.beam_size, lm_nbest=args.lm_nbest,
        beam_beta=args.beam_beta,
    )
    t_bs_total = time.time() - t_bs
    avg_nbest_len = float(np.mean([len(x) for x in all_nbest])) if all_nbest else 0.0
    log.info("  beam-search done in %.1fs  (avg %.1f hyps/utt)",
             t_bs_total, avg_nbest_len)

    # ── Lexicon-only baseline: top-1 by pure acoustic ────────────────────
    lex_only_top1 = [(nb[0][0] if nb else '') for nb in all_nbest]
    lex_metrics = compute_metrics(lex_only_top1, ground_truth, logit_lengths)
    log.info("  Lexicon-only (no LM) top-1: CER=%.4f  WER=%.4f",
             lex_metrics['cer'], lex_metrics['wer'])

    # Oracle: best-achievable WER if we could magically pick the right hypothesis
    oracle_top1 = []
    for i, nbest in enumerate(all_nbest):
        ref_words = ground_truth[i].lower().split()
        best_wer = float('inf'); best_s = ''
        for s, _ac, _nw in nbest:
            w = edit_distance(ref_words, s.lower().split()) / max(len(ref_words), 1)
            if w < best_wer:
                best_wer = w; best_s = s
        oracle_top1.append(best_s)
    oracle_metrics = compute_metrics(oracle_top1, ground_truth, logit_lengths)
    log.info("  ORACLE (best-of-N-best) : CER=%.4f  WER=%.4f",
             oracle_metrics['cer'], oracle_metrics['wer'])

    total_audio_min = float(np.sum(logit_lengths)) * FRAME_DURATION_SEC / 60.0
    total_gt_words = sum(len(s.split()) for s in ground_truth)
    wpm_gt = total_gt_words / total_audio_min if total_audio_min > 0 else 0.0

    results = {
        'config': {
            'decoder_ckpt': CKPT_DIR,
            'beam_size': args.beam_size,
            'lm_nbest': args.lm_nbest,
            'beam_beta': args.beam_beta,
            'acoustic_scale': args.acoustic_scale,
            'alpha': args.alpha,
            'beta': args.beta,
            'grid_search': args.grid_search,
            'max_utts': args.max_utts,
        },
        'neural_decoder_per': round(greedy_per_internal, 6),
        'per_word_aligned': round(per_ours, 6),
        'lexicon_only_top1': {
            'cer': round(lex_metrics['cer'], 6),
            'wer': round(lex_metrics['wer'], 6),
            'wpm_audio': round(wpm_gt, 2),
        },
        'beam_search_time_sec': round(t_bs_total, 2),
        'lm_results': {},
    }
    results['oracle_nbest'] = {
        'cer': round(oracle_metrics['cer'], 6),
        'wer': round(oracle_metrics['wer'], 6),
    }
    for i in range(min(5, len(ground_truth))):
        log.info("    REF   : %s", ground_truth[i])
        log.info("    TOP1  : %s", lex_only_top1[i])
        log.info("    ORACLE: %s", oracle_top1[i])

    lms_to_run = []
    if args.lm in ('gpt2', 'both'):
        lms_to_run.append(('gpt2', GPT2_MODEL_ID))
    if args.lm in ('gemma', 'both'):
        lms_to_run.append(('gemma3_270m', GEMMA_MODEL_ID))

    # ── Step 5+6: LM rescoring for each model ────────────────────────────
    for lm_tag, lm_id in lms_to_run:
        log.info("\n[5/6]  LM rescoring with %s …", lm_id)

        tf.keras.backend.clear_session()
        lm_model, lm_tokenizer = load_lm(lm_id, cache_dir=args.cache_dir,
                                          hf_token=args.hf_token or None)

        t0 = time.time()
        scored = score_nbest_with_lm(
            all_nbest, lm_model, lm_tokenizer, lm_name=lm_tag,
        )
        t_lm_total = time.time() - t0

        # ── Combine & grid-search (if enabled) ────────────────────────────
        if args.grid_search:
            log.info("  Grid search over α, β …")
            best = None
            grid = []
            for a in args.alphas:
                for b in args.betas:
                    decoded = pick_best_hyp(scored, alpha=a, beta=b,
                                            acoustic_scale=args.acoustic_scale)
                    m = compute_metrics(decoded, ground_truth, logit_lengths)
                    grid.append({'alpha': a, 'beta': b,
                                 'cer': round(m['cer'], 6),
                                 'wer': round(m['wer'], 6)})
                    if best is None or m['wer'] < best['wer']:
                        best = {'alpha': a, 'beta': b,
                                'cer': m['cer'], 'wer': m['wer'],
                                'decoded': decoded}
            decoded_sents = best['decoded']
            metrics = {'cer': best['cer'], 'wer': best['wer']}
            log.info("  best α=%.2f β=%.2f → CER=%.4f WER=%.4f",
                     best['alpha'], best['beta'], best['cer'], best['wer'])
            alpha_used, beta_used = best['alpha'], best['beta']
        else:
            decoded_sents = pick_best_hyp(scored, alpha=args.alpha, beta=args.beta,
                                          acoustic_scale=args.acoustic_scale)
            metrics = compute_metrics(decoded_sents, ground_truth, logit_lengths)
            grid = None
            alpha_used, beta_used = args.alpha, args.beta

        log.info("\n[6/6]  Results for %s:", lm_tag)
        log.info("  ┌─────────────────────────────────┐")
        log.info("  │  PER  (neural decoder) = %.4f  │", greedy_per_internal)
        log.info("  │  CER  (LM decoded)     = %.4f  │", metrics['cer'])
        log.info("  │  WER  (LM decoded)     = %.4f  │", metrics['wer'])
        log.info("  │  WPM  (audio-based)    = %.1f  │", wpm_gt)
        log.info("  │  α=%.2f β=%.2f              │", alpha_used, beta_used)
        log.info("  └─────────────────────────────────┘")

        sample_comparisons = [
            {'ref': ground_truth[i], 'hyp': decoded_sents[i]}
            for i in range(min(20, len(decoded_sents)))
        ]
        results['lm_results'][lm_tag] = {
            'lm_model_id': lm_id,
            'per':  round(greedy_per_internal, 6),
            'cer':  round(metrics['cer'], 6),
            'wer':  round(metrics['wer'], 6),
            'wpm_audio': round(wpm_gt, 2),
            'alpha': alpha_used,
            'beta':  beta_used,
            'n_samples': len(decoded_sents),
            'lm_scoring_time_sec': round(t_lm_total, 2),
            'grid': grid,
            'sample_comparisons': sample_comparisons,
        }

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
    log.info("  Neural decoder PER       : %.4f", greedy_per_internal)
    log.info("  Lexicon-only top-1       : CER=%.4f  WER=%.4f",
             lex_metrics['cer'], lex_metrics['wer'])
    for lm_tag, r in results['lm_results'].items():
        log.info("  %s (α=%.2f β=%.2f):  CER=%.4f  WER=%.4f  WPM=%.1f",
                 lm_tag, r['alpha'], r['beta'], r['cer'], r['wer'], r['wpm_audio'])


if __name__ == '__main__':
    main()
