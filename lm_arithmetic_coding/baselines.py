"""
baselines.py
------------
Baseline compression methods for comparison:

  1. Shannon entropy — theoretical lower bound (static byte model)
  2. Static Huffman — character-frequency Huffman tree
  3. Static Arithmetic Coding — character-frequency arithmetic coding
     (uses our ArithmeticEncoder/Decoder)
  4. gzip -9        — stdlib, LZ77 + Huffman
  5. bzip2 -9       — stdlib, BWT + Huffman (typically beats gzip on text)
  6. lzma preset=9  — stdlib, LZMA2/XZ (best general-purpose stdlib compressor)

All methods report bits-per-character (bpc) on a given text.
"""

import bz2
import lzma
import math
import heapq
import gzip
import struct
from collections import Counter
from arithmetic_coder import (
    ArithmeticEncoder, ArithmeticDecoder,
    build_cdf, bits_to_bytes, bytes_to_bits
)


# ---------------------------------------------------------------------------
# 1. gzip baseline
# ---------------------------------------------------------------------------

def gzip_bpc(text: bytes) -> float:
    """Compress with gzip (level 9) and return bits-per-character."""
    compressed = gzip.compress(text, compresslevel=9)
    return (len(compressed) * 8) / len(text)


# ---------------------------------------------------------------------------
# 2. bzip2 baseline
# ---------------------------------------------------------------------------

def bzip2_bpc(text: bytes) -> float:
    """Compress with bzip2 (compresslevel=9) and return bits-per-character.

    bzip2 uses Burrows-Wheeler Transform (BWT) + Move-To-Front + Huffman.
    It typically achieves 10-15% better compression than gzip on natural text.
    """
    compressed = bz2.compress(text, compresslevel=9)
    return (len(compressed) * 8) / len(text)


# ---------------------------------------------------------------------------
# 3. lzma baseline
# ---------------------------------------------------------------------------

def lzma_bpc(text: bytes) -> float:
    """Compress with LZMA (preset=9) and return bits-per-character.

    LZMA2/XZ is the strongest general-purpose stdlib compressor —
    uses a large dictionary + range coder. Closest stdlib competitor to LLM-AC.
    """
    compressed = lzma.compress(text, preset=9)
    return (len(compressed) * 8) / len(text)


# ---------------------------------------------------------------------------
# 4. Static Huffman
# ---------------------------------------------------------------------------

class HuffmanNode:
    def __init__(self, symbol, freq, left=None, right=None):
        self.symbol = symbol
        self.freq   = freq
        self.left   = left
        self.right  = right

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(freq: dict) -> HuffmanNode:
    heap = [HuffmanNode(s, f) for s, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, HuffmanNode(None, a.freq + b.freq, a, b))
    return heap[0]


def build_codebook(node: HuffmanNode, prefix="", codebook=None) -> dict:
    if codebook is None:
        codebook = {}
    if node.symbol is not None:
        codebook[node.symbol] = prefix or "0"
    else:
        build_codebook(node.left,  prefix + "0", codebook)
        build_codebook(node.right, prefix + "1", codebook)
    return codebook


def huffman_bpc(text: bytes) -> tuple:
    """
    Encode `text` with static Huffman coding.
    Returns (bpc, compressed_bytes, codebook).
    """
    freq     = Counter(text)
    tree     = build_huffman_tree(freq)
    codebook = build_codebook(tree)

    # Encode
    bitstring = "".join(codebook[b] for b in text)
    n_bits    = len(bitstring)
    bpc       = n_bits / len(text)

    # Pack into bytes (for size measurement only)
    padded = bitstring + "0" * (-len(bitstring) % 8)
    compressed = bytes(
        int(padded[i:i+8], 2) for i in range(0, len(padded), 8)
    )
    return bpc, compressed, codebook


# ---------------------------------------------------------------------------
# 5. Static Arithmetic Coding (character level)
# ---------------------------------------------------------------------------

def static_arithmetic_bpc(text: bytes) -> tuple:
    """
    Encode `text` with arithmetic coding using a static character-frequency
    probability model (Laplace-smoothed to handle unseen bytes).
    Returns (bpc, compressed_bytes).
    """
    # Build frequency table over all 256 possible bytes
    freq = [0] * 256
    for b in text:
        freq[b] += 1

    cdf, total = build_cdf(freq)

    # Encode
    enc = ArithmeticEncoder()
    for b in text:
        enc.encode(cdf[b], cdf[b + 1], total)
    bits       = enc.finish()
    compressed = bits_to_bytes(bits)
    bpc        = len(bits) / len(text)

    return bpc, compressed


def static_arithmetic_decode(compressed: bytes, n_symbols: int, freq: list) -> bytes:
    """Decode bytes produced by static_arithmetic_encode."""
    cdf, total = build_cdf(freq)
    bits = bytes_to_bits(compressed)
    dec  = ArithmeticDecoder(bits)
    return bytes(dec.decode(cdf, total) for _ in range(n_symbols))


# ---------------------------------------------------------------------------
# Entropy lower bound
# ---------------------------------------------------------------------------

def entropy_bpc(text: bytes) -> float:
    """
    Shannon entropy H(X) in bits/char — the theoretical lower bound for any
    lossless compressor using a static character-level model.
    """
    freq  = Counter(text)
    n     = len(text)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_baselines(text: bytes) -> dict:
    """Run all baselines and return a results dict."""
    results = {}

    results["Shannon entropy (lower bound)"] = entropy_bpc(text)

    h_bpc, _, _ = huffman_bpc(text)
    results["Static Huffman"]                = h_bpc

    a_bpc, _    = static_arithmetic_bpc(text)
    results["Static Arithmetic Coding"]      = a_bpc

    results["gzip -9"]                       = gzip_bpc(text)
    results["bzip2 -9"]                      = bzip2_bpc(text)
    results["lzma preset=9"]                 = lzma_bpc(text)

    return results
