"""
test_edge_cases.py
------------------
Edge case tests for BLM6106 Data Compression Project — Approach 1.

Covers:
  0. bits_to_bytes / bytes_to_bits roundtrip integrity
  1. Arithmetic coder correctness — static path  (build_cdf, Laplace-smoothed)
  2. CDF scaling stress tests — LLM path         (probs_to_int_cdf, float probabilities)
  3. LLM pipeline roundtrip                       (probs_to_int_cdf → encode → decode)
  4. Float16 overflow                             (documents the real production bug + fix)
  5. Compression boundary cases                   (entropy gap, random vs. repetitive)
  6. GPT-2 tokenization round-trip                (requires transformers)

Sections 0–5 run with only numpy (no torch needed).
Section 6 runs only if transformers is installed.

Results are saved to results/edge_case_report.txt.

Usage:
    python test_edge_cases.py
"""

import os
import sys
import math
import numpy as np
from collections import Counter

sys.path.insert(0, ".")
from arithmetic_coder import (
    ArithmeticEncoder, ArithmeticDecoder,
    build_cdf, bits_to_bytes, bytes_to_bits,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

log_lines = []


def log(line: str = ""):
    print(line)
    log_lines.append(line)


# ---------------------------------------------------------------------------
# Standalone CDF converter — mirrors lm_compressor.py exactly
# (no torch import needed; must stay in sync with lm_compressor.py)
# ---------------------------------------------------------------------------

SCALE_TOTAL = 10_000_000
MIN_PROB    = 1e-9
GPT2_VOCAB  = 50_257


def probs_to_int_cdf(probs: np.ndarray):
    """
    Convert a float probability array to an integer CDF suitable for the
    arithmetic coder.  Mirrors the implementation in lm_compressor.py exactly.
    probs must be dtype float32 (or float64).
    """
    probs  = np.maximum(probs, MIN_PROB)
    probs  = probs / probs.sum()
    counts = np.floor(probs * SCALE_TOTAL).astype(np.int64)

    deficit = int(SCALE_TOTAL - counts.sum())
    if deficit > 0:
        top_idx = np.argsort(probs)[::-1][:deficit]
        counts[top_idx] += 1
    elif deficit < 0:
        top_idx = np.argsort(counts)[::-1][:-deficit]
        counts[top_idx] -= 1

    counts = np.maximum(counts, 1)
    total  = int(counts.sum())

    cdf = [0] * (len(counts) + 1)
    for i, c in enumerate(counts):
        cdf[i + 1] = cdf[i] + int(c)
    return cdf, total


# ---------------------------------------------------------------------------
# Helper: static-path encode + decode and assert losslessness
# (uses build_cdf, which is what baselines.py uses — NOT the LLM path)
# ---------------------------------------------------------------------------

def roundtrip_static(text: bytes, label: str):
    """
    Losslessness test using the static character-frequency path (build_cdf).
    This is the same path used by static_arithmetic_bpc() in baselines.py.
    It is NOT the same as the LLM path (probs_to_int_cdf).
    """
    freq = [0] * 256
    for b in text:
        freq[b] += 1
    cdf, total = build_cdf(freq)

    enc = ArithmeticEncoder()
    for b in text:
        enc.encode(cdf[b], cdf[b + 1], total)
    bits       = enc.finish()
    compressed = bits_to_bytes(bits)

    bits2   = bytes_to_bits(compressed)[:len(bits)]
    dec     = ArithmeticDecoder(bits2)
    decoded = bytes(dec.decode(cdf, total) for _ in range(len(text)))

    ok  = (decoded == text)
    bpc = len(bits) / max(len(text), 1)
    log(f"  {PASS if ok else FAIL}  {label:<50}  {bpc:>6.3f} bpc")
    return ok, bpc


# ---------------------------------------------------------------------------
# 0. bits_to_bytes / bytes_to_bits roundtrip integrity
# ---------------------------------------------------------------------------

def test_bits_roundtrip():
    """
    Verify that bits_to_bytes followed by bytes_to_bits[:n] is an identity.
    This is critical: the decoder always strips trailing padding by slicing
    the decompressed bit list to the original length.  Any byte-packing bug
    here corrupts every decode.
    """
    log("\n=== 0. bits_to_bytes / bytes_to_bits Roundtrip ===")
    rng    = np.random.default_rng(7)
    all_ok = True

    def check(bits, label):
        packed   = bits_to_bytes(bits)
        unpacked = bytes_to_bits(packed)[:len(bits)]
        ok       = (unpacked == bits)
        log(f"  {PASS if ok else FAIL}  {label:<55}")
        return ok

    # Fixed patterns
    all_ok &= check([],                                  "Empty bit list")
    all_ok &= check([0],                                 "Single 0")
    all_ok &= check([1],                                 "Single 1")
    all_ok &= check([1, 0, 1, 0, 1, 0, 1, 0],          "Exactly 8 bits (0xAA)")
    all_ok &= check([1] * 8,                             "8 x 1 (0xFF)")
    all_ok &= check([0] * 8,                             "8 x 0 (0x00)")
    all_ok &= check([1, 0, 1, 1, 0, 0, 1],              "7 bits (non-multiple)")
    all_ok &= check([1, 0, 1, 1, 0, 0, 1, 0, 1],       "9 bits (non-multiple)")
    all_ok &= check([0, 1] * 8,                          "16 bits alternating")
    all_ok &= check([1, 1, 0, 0] * 4,                   "16 bits 1100 pattern")

    # Random sequences of various lengths
    for n in [1, 7, 8, 9, 15, 16, 31, 32, 100, 1000, 9999]:
        bits = rng.integers(0, 2, size=n).tolist()
        ok   = check(bits, f"Random {n} bits")
        all_ok &= ok

    # Verify packed byte count formula: ceil(n/8) bytes
    for n in [0, 1, 7, 8, 9, 100]:
        bits  = [1] * n
        packed = bits_to_bytes(bits)
        expected_bytes = (n + 7) // 8 if n > 0 else 0
        ok = (len(packed) == expected_bytes)
        log(f"  {PASS if ok else FAIL}  Byte count for {n} bits: "
            f"expected {expected_bytes}, got {len(packed)}")
        all_ok &= ok

    return all_ok


# ---------------------------------------------------------------------------
# 1. Arithmetic Coder — losslessness (static path)
# ---------------------------------------------------------------------------

def test_arithmetic_coder():
    """
    Test losslessness of ArithmeticEncoder/Decoder using the STATIC path:
    build_cdf() with Laplace smoothing (+1 to every byte count).

    This is the same code path used by static_arithmetic_bpc() in baselines.py.
    It is NOT the LLM compression path; see test_llm_pipeline_roundtrip() for that.
    """
    log("\n=== 1. Arithmetic Coder Losslessness (Static Path — build_cdf) ===")
    import os as _os
    all_ok = True
    cases = [
        (b"a",                                       "Single byte"),
        (b"ab",                                      "Two different bytes"),
        (b"hello, world!",                           "Short ASCII"),
        (b"a" * 1000,                                "1000x repeated char"),
        (b"ab" * 500,                                "Alternating 2 chars (1000 bytes)"),
        (b"the quick brown fox " * 50,               "Repeated phrase (1000 bytes)"),
        (_os.urandom(100),                           "100 random bytes"),
        (_os.urandom(1000),                          "1000 random bytes"),
        (bytes(range(256)),                          "All 256 byte values once"),
        (bytes(range(256)) * 4,                      "All 256 byte values x4"),
        (b"\x00" * 500 + b"\xff" * 500,             "Two-block run"),
        (b"data compression is fascinating! " * 30, "Natural language phrase x30"),
    ]
    for text, label in cases:
        ok, _ = roundtrip_static(text, label)
        all_ok = all_ok and ok
    return all_ok


# ---------------------------------------------------------------------------
# 2. CDF Scaling — stress tests (LLM path)
# ---------------------------------------------------------------------------

def test_cdf_scaling():
    """
    Stress-test probs_to_int_cdf() — the function used by lm_compressor.py —
    with degenerate and extreme float32 probability distributions over the
    full GPT-2 vocabulary (50,257 tokens).

    Checks:
      * No symbol gets zero count (would cause CDF assertion error in encoder)
      * CDF is monotone non-decreasing
      * Dominant-token fraction preserved to >= 99% of its true probability
    """
    log("\n=== 2. CDF Scaling Stress Tests (LLM Path — probs_to_int_cdf) ===")
    all_ok = True

    def check(label, probs):
        cdf, total  = probs_to_int_cdf(probs.astype(np.float32))
        counts      = [cdf[i+1] - cdf[i] for i in range(GPT2_VOCAB)]
        zero_count  = sum(1 for c in counts if c <= 0)
        total_match = (cdf[-1] == total)
        sorted_ok   = all(cdf[i] <= cdf[i+1] for i in range(GPT2_VOCAB))
        ok          = (zero_count == 0) and total_match and sorted_ok
        issues      = []
        if zero_count:      issues.append(f"{zero_count} zero-count symbols")
        if not total_match: issues.append("CDF total mismatch")
        if not sorted_ok:   issues.append("CDF not monotone")
        note = "  Issues: " + ", ".join(issues) if issues else ""
        log(f"  {PASS if ok else FAIL}  {label:<54}{note}")
        return ok

    all_ok &= check("Uniform (all tokens equal)",
                    np.ones(GPT2_VOCAB) / GPT2_VOCAB)
    all_ok &= check("Near-deterministic (one token p=0.9999)",
                    np.concatenate([[0.9999], np.full(GPT2_VOCAB-1, 1e-9/GPT2_VOCAB)]))
    all_ok &= check("Two dominant tokens (p=0.5 each)",
                    np.concatenate([[0.5, 0.5], np.full(GPT2_VOCAB-2, 1e-12)]))
    all_ok &= check("All MIN_PROB (worst-case clamping)",
                    np.full(GPT2_VOCAB, 1e-9))
    all_ok &= check("Geometric distribution",
                    (0.5 ** np.arange(1, GPT2_VOCAB+1) / (1 - 0.5**GPT2_VOCAB)).astype(np.float32))
    rng = np.random.default_rng(42)
    all_ok &= check("Random Dirichlet (alpha=0.1)",
                    rng.dirichlet(np.full(GPT2_VOCAB, 0.1)).astype(np.float32))

    # Dominant-token fraction check
    probs       = np.full(GPT2_VOCAB, 1e-9, dtype=np.float32)
    probs[999]  = 1.0
    cdf, total  = probs_to_int_cdf(probs)
    dom_frac    = (cdf[1000] - cdf[999]) / total
    ok          = dom_frac > 0.99
    all_ok     &= ok
    log(f"  {PASS if ok else FAIL}  Dominant token fraction = {dom_frac:.6f}  (expected > 0.99)")

    return all_ok


# ---------------------------------------------------------------------------
# 3. LLM Pipeline Roundtrip
# ---------------------------------------------------------------------------

def test_llm_pipeline_roundtrip():
    """
    End-to-end losslessness test through the ACTUAL LLM compression path:
      probs_to_int_cdf (float32) -> ArithmeticEncoder -> bits_to_bytes
                                 -> bytes_to_bits -> ArithmeticDecoder

    This is the code path executed by lm_compressor.py for every token.
    Section 1 does NOT cover this path (it uses build_cdf, not probs_to_int_cdf).
    """
    log("\n=== 3. LLM Pipeline Roundtrip (probs_to_int_cdf -> encode -> decode) ===")
    rng    = np.random.default_rng(17)
    all_ok = True

    def encode_decode(token_ids: list, probs_per_step: list, label: str) -> bool:
        enc    = ArithmeticEncoder()
        cdfs   = []
        totals = []
        for i, tid in enumerate(token_ids):
            cdf, total = probs_to_int_cdf(probs_per_step[i])
            cdfs.append(cdf)
            totals.append(total)
            enc.encode(cdf[tid], cdf[tid + 1], total)
        bits       = enc.finish()
        compressed = bits_to_bytes(bits)

        bits2   = bytes_to_bits(compressed)[:len(bits)]
        dec     = ArithmeticDecoder(bits2)
        decoded = [dec.decode(cdfs[i], totals[i]) for i in range(len(token_ids))]

        ok = (decoded == token_ids)
        log(f"  {PASS if ok else FAIL}  {label:<60}")
        if not ok:
            log(f"         Expected: {token_ids[:10]}{'...' if len(token_ids)>10 else ''}")
            log(f"         Got:      {decoded[:10]}{'...' if len(decoded)>10 else ''}")
        return ok

    def uniform_steps(n, vocab=GPT2_VOCAB):
        p = np.ones(vocab, dtype=np.float32) / vocab
        return [p] * n

    def peaked_steps(n, peak_idx, peak_val=0.9, vocab=GPT2_VOCAB):
        base = (1.0 - peak_val) / (vocab - 1)
        p    = np.full(vocab, base, dtype=np.float32)
        p[peak_idx] = peak_val
        return [p] * n

    def random_steps(n, rng, vocab=GPT2_VOCAB):
        return [rng.dirichlet(np.ones(vocab)).astype(np.float32) for _ in range(n)]

    all_ok &= encode_decode([0],             uniform_steps(1),             "Single token (id=0), uniform")
    all_ok &= encode_decode([GPT2_VOCAB-1],  uniform_steps(1),             "Single token (id=50256), uniform")
    all_ok &= encode_decode([999],           peaked_steps(1, 999, 0.9999), "Single token, near-deterministic")

    ids_10 = rng.integers(0, GPT2_VOCAB, size=10).tolist()
    all_ok &= encode_decode(ids_10, uniform_steps(10),              "10 tokens, uniform distribution")
    all_ok &= encode_decode(ids_10, random_steps(10, rng),          "10 tokens, random Dirichlet")

    ids_10_b = rng.integers(0, GPT2_VOCAB, size=10).tolist()
    all_ok &= encode_decode(ids_10_b,
                            [peaked_steps(1, t)[0] for t in ids_10_b],
                            "10 tokens, model perfectly predicts each token")

    ids_100 = rng.integers(0, GPT2_VOCAB, size=100).tolist()
    all_ok &= encode_decode(ids_100, uniform_steps(100),            "100 tokens, uniform")
    all_ok &= encode_decode(ids_100, random_steps(100, rng),        "100 tokens, random Dirichlet")

    same_ids = [42] * 20
    all_ok &= encode_decode(same_ids, uniform_steps(20),            "20x same token (id=42), uniform")
    all_ok &= encode_decode(same_ids,
                            [peaked_steps(1, 42, 0.99)[0]] * 20,
                            "20x same token (id=42), peaked at 42")

    return all_ok


# ---------------------------------------------------------------------------
# 4. Float16 Overflow
# ---------------------------------------------------------------------------

def test_float16_overflow():
    """
    Documents and tests the float16 overflow bug found in lm_compressor.py.

    Root cause: GPT-2 loaded with torch_dtype=float16.  torch.softmax on float16
    logits returns float16 probabilities.  For any token with p > 0.0066,
    p * SCALE_TOTAL (= 10,000,000) exceeds float16 max (65,504) -> inf.
    np.floor(inf).astype(np.int64) produces INT64_MAX, corrupting the CDF and
    triggering the encoder assertion:
      "Invalid CDF: [105432365753...) / 9223372036854725551"

    Fix applied in lm_compressor.py:
      # BEFORE (broken):
      probs = torch.softmax(out.logits[0, -1], dim=-1).cpu().numpy()
      # AFTER (fixed):
      probs = torch.softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()

    Tests A-D below confirm the overflow exists in float16 and is absent in float32.
    """
    log("\n=== 4. Float16 Overflow — Production Bug & Fix ===")
    all_ok = True

    FLOAT16_MAX  = float(np.finfo(np.float16).max)   # 65504.0
    OVERFLOW_THR = FLOAT16_MAX / SCALE_TOTAL          # ~0.006550

    log(f"  Float16 max        : {FLOAT16_MAX}")
    log(f"  SCALE_TOTAL        : {SCALE_TOTAL}")
    log(f"  Overflow threshold : p > {OVERFLOW_THR:.6f} causes inf in float16 path")
    log("")

    # A: float16 path produces inf for a realistic dominant token (p=0.9)
    probs_f16        = np.zeros(GPT2_VOCAB, dtype=np.float16)
    probs_f16[0]     = np.float16(0.9)
    probs_f16[1:]    = np.float16(1e-5)
    clamped_f16      = np.maximum(probs_f16, np.float16(MIN_PROB))
    normed_f16       = (clamped_f16 / clamped_f16.sum()).astype(np.float16)
    scaled_f16       = normed_f16 * np.float16(SCALE_TOTAL)
    has_inf_f16      = bool(np.any(np.isinf(scaled_f16)))
    ok_A             = has_inf_f16
    all_ok          &= ok_A
    log(f"  {PASS if ok_A else FAIL}  Float16 path produces inf for p(token_0)=0.9"
        f"  [has_inf={has_inf_f16}]")

    # B: float32 cast avoids overflow for the same distribution
    probs_f32   = probs_f16.astype(np.float32)
    clamped_f32 = np.maximum(probs_f32, MIN_PROB)
    normed_f32  = clamped_f32 / clamped_f32.sum()
    scaled_f32  = normed_f32 * SCALE_TOTAL
    has_inf_f32 = bool(np.any(np.isinf(scaled_f32)))
    ok_B        = not has_inf_f32
    all_ok     &= ok_B
    log(f"  {PASS if ok_B else FAIL}  Float32 path has NO inf for same distribution"
        f"           [has_inf={has_inf_f32}]")

    # C: probs_to_int_cdf on float32 produces a valid CDF
    cdf, total = probs_to_int_cdf(probs_f32)
    counts     = [cdf[i+1] - cdf[i] for i in range(GPT2_VOCAB)]
    zero_count = sum(1 for c in counts if c <= 0)
    sorted_ok  = all(cdf[i] <= cdf[i+1] for i in range(GPT2_VOCAB))
    ok_C       = (zero_count == 0) and (cdf[-1] == total) and sorted_ok
    all_ok    &= ok_C
    log(f"  {PASS if ok_C else FAIL}  probs_to_int_cdf(float32) -> valid CDF"
        f"                 [zero_counts={zero_count}, sorted={sorted_ok}]")

    # D: float32 CDF valid across a range of dominant-token probabilities
    for p_val in [0.0065, 0.0066, 0.01, 0.1, 0.5, 0.99]:
        p    = np.full(GPT2_VOCAB, (1 - p_val) / (GPT2_VOCAB - 1), dtype=np.float32)
        p[0] = p_val
        cdf_d, total_d = probs_to_int_cdf(p)
        ok_d = (cdf_d[0] == 0) and (cdf_d[-1] == total_d) and (cdf_d[1] > cdf_d[0])
        all_ok &= ok_d
        log(f"  {PASS if ok_d else FAIL}  float32 CDF valid for dominant p={p_val:.4f}")

    return all_ok


# ---------------------------------------------------------------------------
# 5. Compression boundary cases
# ---------------------------------------------------------------------------

def test_compression_boundaries():
    """
    Verify that the static arithmetic coder achieves expected BPC ranges on
    extremal inputs.  All three sub-cases emit explicit [PASS]/[FAIL].
    """
    log("\n=== 5. Compression Boundary Cases ===")
    import os as _os
    all_ok = True

    # 5a. Near-zero BPC — highly repetitive input
    # build_cdf adds Laplace smoothing (+1 to all 256 byte counts).
    # Overhead ~ log2(257) * 256/N; for N=10,000 ~ 0.21 bpc, threshold = 0.25.
    text    = b"a" * 10_000
    ok, bpc = roundtrip_static(text, "10 000x 'a'  (expect BPC < 0.25)")
    ok_a    = ok and (bpc < 0.25)
    all_ok &= ok_a
    log(f"  {PASS if ok_a else FAIL}  BPC = {bpc:.5f}  "
        f"({'ok' if bpc < 0.25 else 'exceeded 0.25 — check Laplace overhead'})")

    # 5b. Near-8 BPC — random bytes (maximum entropy input)
    text    = _os.urandom(10_000)
    ok, bpc = roundtrip_static(text, "10 000 random bytes  (expect BPC > 7.9)")
    ok_b    = ok and (bpc > 7.9)
    all_ok &= ok_b
    log(f"  {PASS if ok_b else FAIL}  BPC = {bpc:.4f}  "
        f"({'ok' if bpc > 7.9 else 'suspiciously low'})")

    # 5c. Shannon entropy gap — natural language phrase
    # Static AC should be within 0.1 bpc of Shannon entropy.
    text    = b"the quick brown fox jumps over the lazy dog. " * 200
    freq    = Counter(text)
    n       = len(text)
    h       = -sum((c/n) * math.log2(c/n) for c in freq.values())
    ok, bpc = roundtrip_static(text, "Phrase x200 — entropy gap check")
    gap     = bpc - h
    ok_c    = ok and (gap < 0.1)
    all_ok &= ok_c
    log(f"  {PASS if ok_c else FAIL}  H(X)={h:.4f}  Actual={bpc:.4f}  Gap={gap:.5f}  "
        f"({'ok' if ok_c else 'gap > 0.1'})")

    return all_ok


# ---------------------------------------------------------------------------
# 6. GPT-2 tokenization round-trip
# ---------------------------------------------------------------------------

def test_tokenization():
    """
    Verify that GPT-2 tokenizer encodes and decodes without loss for clean text,
    AND that all returned token IDs are within the valid vocabulary range [0, 50256].
    Non-ASCII strings are expected to be lossy — noted, not failed.
    """
    log("\n=== 6. GPT-2 Tokenization Round-Trip ===")
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from transformers import GPT2TokenizerFast
            tok = GPT2TokenizerFast.from_pretrained("gpt2")
    except ImportError:
        log(f"  {SKIP}  transformers not installed")
        return

    VOCAB_SIZE = 50_257
    all_ok     = True

    samples = [
        ("Clean prose",   "The history of the Roman Empire is fascinating.", True),
        ("XML markup",    "<page><title>Albert Einstein</title></page>",      True),
        ("Numbers",       "In 2023, approximately 8,045,311,447 people lived on Earth.", True),
        ("Punctuation",   "Hello! How are you? I'm fine -- thanks.",          True),
        ("Non-ASCII",     "cafe, naive, resume, Angstrom",                    False),
    ]

    for label, s, expect_exact in samples:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ids       = tok.encode(s)
            recovered = tok.decode(ids)

        # Vocab range check — mandatory for all inputs
        out_of_range = [i for i in ids if not (0 <= i < VOCAB_SIZE)]
        ok_range     = (len(out_of_range) == 0)
        all_ok      &= ok_range

        # Round-trip check — mandatory only for expected-lossless inputs
        exact_match = (s == recovered)
        if expect_exact:
            ok_rt   = exact_match
            all_ok &= ok_rt
        else:
            ok_rt = True

        ok      = ok_range and ok_rt
        rt_note = "exact match" if exact_match else "lossy (expected for non-ASCII)"
        log(f"  {PASS if ok else FAIL}  {label:<20}  {len(ids):>4} tokens  "
            f"{rt_note}  vocab_ok={ok_range}"
            + (f"  out-of-range={out_of_range}" if not ok_range else ""))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    log("=" * 70)
    log("  BLM6106 — Edge Case Test Report")
    log("=" * 70)

    r0 = test_bits_roundtrip()
    r1 = test_arithmetic_coder()
    r2 = test_cdf_scaling()
    r3 = test_llm_pipeline_roundtrip()
    r4 = test_float16_overflow()
    r5 = test_compression_boundaries()
    test_tokenization()

    log("")
    log("=" * 70)
    log(f"  bits_to_bytes/bytes_to_bits roundtrip : {'ALL PASS' if r0 else 'SOME FAILED'}")
    log(f"  Arithmetic coder losslessness (static): {'ALL PASS' if r1 else 'SOME FAILED'}")
    log(f"  CDF scaling stress tests (LLM path)   : {'ALL PASS' if r2 else 'SOME FAILED'}")
    log(f"  LLM pipeline roundtrip                : {'ALL PASS' if r3 else 'SOME FAILED'}")
    log(f"  Float16 overflow detection            : {'ALL PASS' if r4 else 'SOME FAILED'}")
    log(f"  Compression boundary cases            : {'ALL PASS' if r5 else 'SOME FAILED'}")
    log("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "edge_case_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\n[results] Saved -> {report_path}")


if __name__ == "__main__":
    main()
