"""
test_edge_cases.py
------------------
Edge case verification for the PixelCNN arithmetic coding pipeline.

Analog of ../lm_arithmetic_coding/test_edge_cases.py — same arithmetic coder,
image-specific test cases.  Runs entirely without a trained model; tests only
the arithmetic coder and CDF scaling logic.

Categories:
  1. Uniform 256-value pixel distribution
  2. Near-deterministic distribution (one dominant pixel value)
  3. Degenerate images (all-black, all-white)
  4. Full 28×28 image roundtrip (synthetic, uniform model)
  5. CDF integrity checks across distribution types
  6. BPP compression bounds sanity checks
"""

import os
import sys
import math
import numpy as np

# Reuse arithmetic coder from the text experiment
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "lm_arithmetic_coding"))
from arithmetic_coder import ArithmeticEncoder, ArithmeticDecoder, bits_to_bytes, bytes_to_bits

RESULTS_DIR = os.path.join(_HERE, "results")
SCALE_TOTAL = 10_000_000
MIN_PROB    = 1e-9

passed = 0
failed = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def probs_to_int_cdf(probs: np.ndarray):
    """Standalone copy — no torch dependency for tests."""
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
    cdf    = [0] * 257
    for i, c in enumerate(counts):
        cdf[i + 1] = cdf[i] + int(c)
    return cdf, total


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}  {detail}")
        failed += 1


def roundtrip(symbols, cdfs_totals) -> list:
    """Encode then decode a symbol sequence and return decoded symbols."""
    enc = ArithmeticEncoder()
    for sym, (cdf, total) in zip(symbols, cdfs_totals):
        enc.encode(cdf[sym], cdf[sym + 1], total)
    bits = enc.finish()
    dec  = ArithmeticDecoder(bits)
    return [dec.decode(cdf, total) for cdf, total in cdfs_totals]


# ---------------------------------------------------------------------------
# Category 1: Uniform distribution over all 256 pixel values
# ---------------------------------------------------------------------------
print("\n=== Category 1: Uniform 256-value pixel distribution ===")

probs_u      = np.ones(256) / 256.0
cdf_u, tot_u = probs_to_int_cdf(probs_u)

# Roundtrip across a representative spread of pixel values
symbols  = list(range(0, 256, 16))   # 0, 16, 32, ..., 240  (16 values)
decoded  = roundtrip(symbols, [(cdf_u, tot_u)] * len(symbols))
check("Roundtrip: 16 spread pixel values, uniform model", symbols == decoded)

# Individual boundary values
for pv in [0, 1, 127, 128, 254, 255]:
    enc = ArithmeticEncoder()
    enc.encode(cdf_u[pv], cdf_u[pv + 1], tot_u)
    dec = ArithmeticDecoder(enc.finish())
    check(f"Single pixel roundtrip: value={pv}", dec.decode(cdf_u, tot_u) == pv)


# ---------------------------------------------------------------------------
# Category 2: Near-deterministic distribution (one dominant pixel value)
# ---------------------------------------------------------------------------
print("\n=== Category 2: Near-deterministic distribution ===")

probs_det        = np.ones(256) * (0.001 / 255)
probs_det[200]   = 0.999
cdf_det, tot_det = probs_to_int_cdf(probs_det)

frac = (cdf_det[201] - cdf_det[200]) / tot_det
check(f"Dominant pixel fraction ≥ 0.990  (got {frac:.4f})", frac >= 0.990)

decoded = roundtrip([200, 0, 200, 255, 200],
                    [(cdf_det, tot_det)] * 5)
check("Roundtrip: dominant + rare values mixed", decoded == [200, 0, 200, 255, 200])


# ---------------------------------------------------------------------------
# Category 3: Degenerate images (all-black, all-white)
# ---------------------------------------------------------------------------
print("\n=== Category 3: Degenerate images ===")

for pv, label in [(0, "all-black"), (255, "all-white")]:
    probs_d      = np.zeros(256); probs_d[pv] = 1.0
    cdf_d, tot_d = probs_to_int_cdf(probs_d)
    symbols_d    = [pv] * 784
    decoded_d    = roundtrip(symbols_d, [(cdf_d, tot_d)] * 784)
    check(f"784-pixel {label} image roundtrip (pv={pv})", decoded_d == symbols_d)

# Gradient image: pixel value == pixel index mod 256
probs_g      = np.ones(256) / 256
cdf_g, tot_g = probs_to_int_cdf(probs_g)
gradient     = [i % 256 for i in range(784)]
decoded_g    = roundtrip(gradient, [(cdf_g, tot_g)] * 784)
check("784-pixel gradient image roundtrip (uniform model)", decoded_g == gradient)


# ---------------------------------------------------------------------------
# Category 4: Full 28×28 image roundtrip with per-pixel distributions
# ---------------------------------------------------------------------------
print("\n=== Category 4: Full image roundtrip with varying distributions ===")

np.random.seed(42)
img_flat = np.random.randint(0, 256, 784).tolist()

# Each pixel gets a slightly different distribution (simulate PixelCNN output)
enc = ArithmeticEncoder()
cdfs_for_decode = []
for idx, pv in enumerate(img_flat):
    # Random Dirichlet distribution per pixel
    alpha        = np.random.dirichlet(np.ones(256) * 0.5)
    cdf_i, tot_i = probs_to_int_cdf(alpha)
    enc.encode(cdf_i[pv], cdf_i[pv + 1], tot_i)
    cdfs_for_decode.append((cdf_i, tot_i))

bits = enc.finish()
dec  = ArithmeticDecoder(bits)
decoded_full = [dec.decode(cdf, tot) for cdf, tot in cdfs_for_decode]
check("Full 784-pixel roundtrip with per-pixel Dirichlet distributions",
      decoded_full == img_flat)


# ---------------------------------------------------------------------------
# Category 5: CDF structural integrity
# ---------------------------------------------------------------------------
print("\n=== Category 5: CDF structural integrity ===")

np.random.seed(7)
test_distributions = {
    "uniform 256":       np.ones(256) / 256,
    "Dirichlet α=1":     np.random.dirichlet(np.ones(256)),
    "Dirichlet α=0.1":   np.random.dirichlet(np.ones(256) * 0.1),
    "near-deterministic": probs_det,
    "all-black prior":   probs_det * 0 + np.eye(256)[0],
}

for name, probs in test_distributions.items():
    cdf, total = probs_to_int_cdf(probs)
    check(f"CDF[-1] == total          ({name})", cdf[-1] == total)
    check(f"All counts ≥ 1            ({name})",
          all(cdf[i + 1] - cdf[i] >= 1 for i in range(256)))
    check(f"Total ≥ SCALE_TOTAL       ({name})", total >= SCALE_TOTAL)
    check(f"CDF is non-decreasing     ({name})",
          all(cdf[i + 1] >= cdf[i] for i in range(256)))


# ---------------------------------------------------------------------------
# Category 6: BPP bounds
# ---------------------------------------------------------------------------
print("\n=== Category 6: BPP compression bounds ===")

# Uniform model on a spread of pixel values → should give ~8 bpp
probs_uni      = np.ones(256) / 256
cdf_uni, tot_u = probs_to_int_cdf(probs_uni)
enc_uni = ArithmeticEncoder()
symbols_uni = list(range(256))        # exactly one of each pixel value
for pv in symbols_uni:
    enc_uni.encode(cdf_uni[pv], cdf_uni[pv + 1], tot_u)
bpp_uni = len(enc_uni.finish()) / 256
check(f"Uniform model BPP ≈ 8.0  (got {bpp_uni:.2f})", 7.5 < bpp_uni < 9.0)

# Near-perfect model on a single repeated value → should give < 0.01 bpp
probs_perf        = np.zeros(256); probs_perf[128] = 1.0
cdf_perf, tot_pf  = probs_to_int_cdf(probs_perf)
enc_perf = ArithmeticEncoder()
for _ in range(784):
    enc_perf.encode(cdf_perf[128], cdf_perf[129], tot_pf)
bpp_perf = len(enc_perf.finish()) / 784
check(f"Near-perfect model BPP < 0.1  (got {bpp_perf:.4f})", bpp_perf < 0.1)

# PixelCNN should always do better than raw (8 bpp) on natural images
# Proxy: a Dirichlet(α=0.5) model should beat uniform
probs_dir      = np.random.dirichlet(np.ones(256) * 0.5)
cdf_dir, tot_d = probs_to_int_cdf(probs_dir)
dominant_pv    = int(np.argmax(probs_dir))
enc_dir = ArithmeticEncoder()
for _ in range(100):
    enc_dir.encode(cdf_dir[dominant_pv], cdf_dir[dominant_pv + 1], tot_d)
bpp_dir = len(enc_dir.finish()) / 100
check(f"Skewed model BPP < 8.0 on dominant symbol  (got {bpp_dir:.3f})", bpp_dir < 8.0)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total_tests = passed + failed
print(f"\n{'=' * 52}")
print(f"Results : {passed}/{total_tests} passed  |  {failed} failed")
print(f"{'=' * 52}\n")

os.makedirs(RESULTS_DIR, exist_ok=True)
report_path = os.path.join(RESULTS_DIR, "edge_case_report.txt")
with open(report_path, "w") as f:
    f.write("PixelCNN Arithmetic Coding — Edge Case Verification\n")
    f.write("=" * 52 + "\n")
    f.write(f"Tests passed : {passed}/{total_tests}\n")
    f.write(f"Tests failed : {failed}/{total_tests}\n\n")
    if failed == 0:
        f.write("All tests passed.\n"
                "Arithmetic coder and CDF scaler verified for image domain.\n")
    else:
        f.write(f"WARNING: {failed} test(s) failed — review output above.\n")

print(f"Report saved → {report_path}")

if failed > 0:
    raise SystemExit(1)
