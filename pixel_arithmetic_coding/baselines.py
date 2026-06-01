"""
baselines.py
------------
Baseline image compressors for Fashion MNIST.

Analog of ../lm_arithmetic_coding/baselines.py — same role, image domain.

All functions report bits-per-pixel (BPP) on a single 28×28 grayscale image.
run_baselines_batch() averages over a collection of images.

Baseline ladder (expected order on Fashion MNIST):
  Shannon entropy  ~4.5 bpp  — theoretical floor for static pixel-level model
  gzip -9          ~6.5 bpp  — LZ77; poor on images (no spatial model)
  PNG              ~4.8 bpp  — filter + DEFLATE; the standard lossless baseline
  PixelCNN-AC      ~3.7 bpp  — neural autoregressive model (our method)
"""

import io
import gzip
import math
import numpy as np
from collections import Counter
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Individual baselines
# ---------------------------------------------------------------------------

def raw_bpp(image: np.ndarray) -> float:
    """Uncompressed storage: always 8.0 BPP (1 byte per pixel)."""
    return 8.0


def entropy_bpp(image: np.ndarray) -> float:
    """
    Shannon entropy H(X) in bits/pixel over the empirical pixel distribution.
    Theoretical lower bound for any lossless compressor using a static
    pixel-frequency model — the image analogue of entropy_bpc().
    """
    flat = image.flatten().tolist()
    n    = len(flat)
    freq = Counter(flat)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def gzip_bpp(image: np.ndarray) -> float:
    """
    gzip -9 applied to raw pixel bytes in raster order.
    Included for comparability with the text experiment, but note that gzip
    has no spatial model — it only exploits 1D run-length patterns and has
    no understanding of 2D structure.
    """
    raw        = image.astype(np.uint8).tobytes()
    compressed = gzip.compress(raw, compresslevel=9)
    return (len(compressed) * 8) / image.size


def png_bpp(image: np.ndarray) -> float:
    """
    PNG lossless compression (Pillow).

    PNG applies a row-wise prediction filter (e.g. Sub, Up, Average, Paeth)
    before DEFLATE compression.  This gives it a simple 2D spatial model,
    making it the strongest standard lossless baseline for grayscale images
    — superior to gzip on raw pixels.  It is the reference point our
    PixelCNN must beat to claim a meaningful result.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow required: pip install Pillow")
    buf = io.BytesIO()
    Image.fromarray(image.astype(np.uint8), mode="L").save(
        buf, format="PNG", optimize=True
    )
    return (buf.tell() * 8) / image.size


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_baselines(image: np.ndarray) -> dict:
    """Run all baselines on a single (28, 28) uint8 image."""
    return {
        "Raw (8 bpp)":           raw_bpp(image),
        "Shannon entropy":       entropy_bpp(image),
        "gzip -9 (raw pixels)":  gzip_bpp(image),
        "PNG (lossless)":        png_bpp(image),
    }


def run_baselines_batch(images: np.ndarray) -> dict:
    """
    Average baselines over a batch of images.

    Parameters
    ----------
    images : np.ndarray, shape (N, 28, 28), dtype uint8

    Returns
    -------
    dict mapping method name → mean BPP across all images
    """
    accum: dict = {}
    for img in tqdm(images, desc="Baselines", unit="img", dynamic_ncols=True):
        for name, bpp in run_baselines(img).items():
            accum.setdefault(name, []).append(bpp)
    return {name: float(np.mean(vals)) for name, vals in accum.items()}
