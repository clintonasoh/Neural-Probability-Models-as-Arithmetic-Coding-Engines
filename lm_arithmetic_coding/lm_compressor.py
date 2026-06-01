"""
lm_compressor.py
----------------
LLM-guided arithmetic coding compressor (Approach 1).

Key improvements:
- KV-cache (past_key_values) for O(n) incremental inference
- tqdm progress bars (work in both terminal and Jupyter)
- DynamicCache compatibility (new transformers) + tuple format (old transformers)
- Suppressed verbose model loading output

Reference:
  Delétang et al., "Language Models are Compression Algorithms", 2023.
  https://arxiv.org/abs/2309.10668
"""

import math
import time
import warnings
import logging
import torch
import numpy as np
from typing import List, Tuple, Dict
from tqdm.auto import tqdm

# Suppress all transformers/torch noise
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

from arithmetic_coder import (
    ArithmeticEncoder, ArithmeticDecoder,
    bits_to_bytes, bytes_to_bits,
)

SCALE_TOTAL = 10_000_000
MIN_PROB    = 1e-9


# ---------------------------------------------------------------------------
# Model loader (singleton)
# ---------------------------------------------------------------------------

_model             = None
_tokenizer         = None
_model_name_loaded = None


def load_model(model_name: str = "gpt2"):
    global _model, _tokenizer, _model_name_loaded
    if _model is None or _model_name_loaded != model_name:
        import transformers
        transformers.logging.set_verbosity_error()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[lm_compressor] Loading {model_name}...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _model     = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        _model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model.to(device)
        _model_name_loaded = model_name
        print(f"[lm_compressor] Model loaded on {device}.")
    return _model, _tokenizer


def get_device():
    return next(_model.parameters()).device


# ---------------------------------------------------------------------------
# KV-cache length helper — handles both DynamicCache and legacy tuple format
# ---------------------------------------------------------------------------

def _cache_len(past_kv) -> int:
    """Return the sequence length stored in a past_key_values cache."""
    if past_kv is None:
        return 0
    if hasattr(past_kv, "get_seq_length"):      # DynamicCache (transformers >= 4.36)
        return past_kv.get_seq_length()
    return past_kv[0][0].shape[2]               # legacy tuple format


# ---------------------------------------------------------------------------
# CDF conversion
# ---------------------------------------------------------------------------

def probs_to_int_cdf(probs: np.ndarray) -> Tuple[List[int], int]:
    """Convert a softmax probability vector to an integer CDF."""
    probs  = np.maximum(probs, MIN_PROB)
    probs  = probs / probs.sum()
    counts = np.floor(probs * SCALE_TOTAL).astype(np.int64)

    deficit = int(SCALE_TOTAL - counts.sum())
    if deficit > 0:
        counts[np.argsort(probs)[::-1][:deficit]] += 1
    elif deficit < 0:
        counts[np.argsort(counts)[::-1][:-deficit]] -= 1

    counts = np.maximum(counts, 1)
    total  = int(counts.sum())

    cdf = [0] * (len(counts) + 1)
    for i, c in enumerate(counts):
        cdf[i + 1] = cdf[i] + int(c)
    return cdf, total


# ---------------------------------------------------------------------------
# Cross-entropy BPC (theoretical) — with KV-cache
# ---------------------------------------------------------------------------

def compute_cross_entropy_bpc(
    text: bytes,
    window: int = 512,
    model_name: str = "gpt2",
) -> Tuple[float, float, Dict]:
    """
    Compute GPT-2 cross-entropy on `text` in bits per character.
    Uses incremental KV-cache inference and tqdm progress bar.

    Returns (bits_per_token, bits_per_char, metrics).
    """
    model, tokenizer = load_model(model_name)
    device = get_device()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        token_ids = tokenizer.encode(text.decode("utf-8", errors="replace"))

    n_tokens       = len(token_ids)
    n_chars        = len(text)
    total_nll      = 0.0
    n_forward_pass = 0
    past_kv        = None
    t_start        = time.time()

    with tqdm(total=n_tokens - 1, desc="Cross-entropy BPC", unit="tok",
              dynamic_ncols=True) as pbar:

        for i in range(n_tokens - 1):
            need_rebuild = _cache_len(past_kv) >= window

            if i == 0:
                ids_t = torch.tensor([[token_ids[0]]], dtype=torch.long, device=device)
                with torch.no_grad():
                    out = model(ids_t, use_cache=True)
                past_kv = out.past_key_values
            elif need_rebuild:
                ctx   = token_ids[max(0, i - window + 1) : i + 1]
                ids_t = torch.tensor([ctx], dtype=torch.long, device=device)
                with torch.no_grad():
                    out = model(ids_t, use_cache=True)
                past_kv = out.past_key_values
            else:
                new_tok = torch.tensor([[token_ids[i]]], dtype=torch.long, device=device)
                with torch.no_grad():
                    out = model(new_tok, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values

            n_forward_pass += 1
            target  = token_ids[i + 1]
            log_p   = torch.log_softmax(out.logits[0, -1], dim=-1)[target].item()
            total_nll -= log_p

            pbar.update(1)
            if (i + 1) % 200 == 0:
                elapsed = time.time() - t_start
                pbar.set_postfix({"tok/s": f"{(i+1)/elapsed:.0f}",
                                  "ETA":   f"{(n_tokens-i-1)/((i+1)/elapsed)/60:.1f}m"})

    elapsed      = time.time() - t_start
    n_scored     = n_tokens - 1
    avg_nll_bits = (total_nll / n_scored) / math.log(2)
    bpc          = avg_nll_bits * (n_scored / n_chars)

    metrics = {
        "n_tokens":          n_tokens,
        "n_chars":           n_chars,
        "n_forward_passes":  n_forward_pass,
        "tokens_per_second": round(n_tokens / elapsed, 2),
        "elapsed_seconds":   round(elapsed, 2),
        "bits_per_token":    round(avg_nll_bits, 6),
        "bits_per_char":     round(bpc, 6),
    }
    return avg_nll_bits, bpc, metrics


# ---------------------------------------------------------------------------
# Full LLM Arithmetic Encoder — with KV-cache
# ---------------------------------------------------------------------------

def lm_encode(
    text: bytes,
    window: int = 512,
    model_name: str = "gpt2",
) -> Tuple[bytes, List[int], int, Dict]:
    """
    Losslessly encode `text` using GPT-2 + arithmetic coding.
    Returns (compressed_bytes, token_ids, n_bits, metrics).
    """
    model, tokenizer = load_model(model_name)
    device = get_device()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        token_ids = tokenizer.encode(text.decode("utf-8", errors="replace"))

    n_tokens       = len(token_ids)
    enc            = ArithmeticEncoder()
    past_kv        = None
    n_forward_pass = 0
    t_start        = time.time()
    vocab_size     = tokenizer.vocab_size
    uniform_cdf    = list(range(vocab_size + 1))

    with tqdm(total=n_tokens, desc="Encoding", unit="tok",
              dynamic_ncols=True) as pbar:

        for i in range(n_tokens):
            if i == 0:
                cdf, total = uniform_cdf, vocab_size
            else:
                if _cache_len(past_kv) >= window:
                    ctx   = token_ids[max(0, i - window) : i]
                    ids_t = torch.tensor([ctx], dtype=torch.long, device=device)
                    with torch.no_grad():
                        out = model(ids_t, use_cache=True)
                    past_kv = out.past_key_values
                else:
                    new_tok = torch.tensor([[token_ids[i - 1]]],
                                           dtype=torch.long, device=device)
                    with torch.no_grad():
                        out = model(new_tok, past_key_values=past_kv, use_cache=True)
                    past_kv = out.past_key_values

                n_forward_pass += 1
                probs      = torch.softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
                cdf, total = probs_to_int_cdf(probs)

            enc.encode(cdf[token_ids[i]], cdf[token_ids[i] + 1], total)
            pbar.update(1)
            if (i + 1) % 200 == 0:
                elapsed = time.time() - t_start
                pbar.set_postfix({"tok/s": f"{(i+1)/elapsed:.0f}"})

    bits       = enc.finish()
    compressed = bits_to_bytes(bits)
    elapsed    = time.time() - t_start

    metrics = {
        "n_tokens":          n_tokens,
        "n_forward_passes":  n_forward_pass,
        "tokens_per_second": round(n_tokens / elapsed, 2),
        "elapsed_seconds":   round(elapsed, 2),
        "compressed_bytes":  len(compressed),
        "n_bits":            len(bits),
    }
    print(f"[encode] {len(compressed):,} bytes ({len(bits):,} bits) "
          f"in {elapsed:.1f}s — {n_tokens/elapsed:.0f} tok/s")
    return compressed, token_ids, len(bits), metrics


# ---------------------------------------------------------------------------
# Full LLM Arithmetic Decoder — with KV-cache
# ---------------------------------------------------------------------------

def lm_decode(
    compressed: bytes,
    token_ids: List[int],
    n_bits: int,
    window: int = 512,
    model_name: str = "gpt2",
) -> Tuple[bytes, Dict]:
    """
    Decode bytes produced by lm_encode.
    Returns (recovered_bytes, metrics).
    """
    model, tokenizer = load_model(model_name)
    device = get_device()

    bits           = bytes_to_bits(compressed)[:n_bits]
    dec            = ArithmeticDecoder(bits)
    n_tokens       = len(token_ids)
    decoded_ids    = []
    past_kv        = None
    n_forward_pass = 0
    t_start        = time.time()
    vocab_size     = tokenizer.vocab_size
    uniform_cdf    = list(range(vocab_size + 1))

    with tqdm(total=n_tokens, desc="Decoding", unit="tok",
              dynamic_ncols=True) as pbar:

        for i in range(n_tokens):
            if i == 0:
                cdf, total = uniform_cdf, vocab_size
            else:
                if _cache_len(past_kv) >= window:
                    ctx   = decoded_ids[max(0, i - window) : i]
                    ids_t = torch.tensor([ctx], dtype=torch.long, device=device)
                    with torch.no_grad():
                        out = model(ids_t, use_cache=True)
                    past_kv = out.past_key_values
                else:
                    new_tok = torch.tensor([[decoded_ids[i - 1]]],
                                           dtype=torch.long, device=device)
                    with torch.no_grad():
                        out = model(new_tok, past_key_values=past_kv, use_cache=True)
                    past_kv = out.past_key_values

                n_forward_pass += 1
                probs      = torch.softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
                cdf, total = probs_to_int_cdf(probs)

            decoded_ids.append(dec.decode(cdf, total))
            pbar.update(1)
            if (i + 1) % 200 == 0:
                elapsed = time.time() - t_start
                pbar.set_postfix({"tok/s": f"{(i+1)/elapsed:.0f}"})

    elapsed = time.time() - t_start
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recovered_text = tokenizer.decode(decoded_ids, skip_special_tokens=False)

    metrics = {
        "n_tokens":          n_tokens,
        "n_forward_passes":  n_forward_pass,
        "tokens_per_second": round(n_tokens / elapsed, 2),
        "elapsed_seconds":   round(elapsed, 2),
    }
    print(f"[decode] done in {elapsed:.1f}s — {n_tokens/elapsed:.0f} tok/s")
    return recovered_text.encode("utf-8", errors="replace"), metrics
