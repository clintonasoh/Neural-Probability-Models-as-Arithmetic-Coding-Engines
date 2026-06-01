"""
pixel_compressor.py
-------------------
PixelCNN-guided arithmetic coding for Fashion MNIST images.

Analog of ../lm_arithmetic_coding/lm_compressor.py — same arithmetic coder,
different probability model.  The arithmetic_coder module is imported directly
from the sibling folder; no code is duplicated.

Note on encode/decode symmetry vs. GPT-2 (text experiment):
  Encoding : 784 sequential forward passes — mirrors the decoder exactly so that
             float32 CDF distributions match bit-for-bit (losslessness requires this).
             A single-pass encode is correct in exact arithmetic but fails in float32
             due to numerical drift across deep residual blocks.
  Decoding : 784 sequential forward passes — each pixel must be decoded before
             it can act as context for the next, so generation is inherently serial.

Both encode and decode cost O(n) forward passes, same as the text experiment
(GPT-2 with KV-cache). The key difference from GPT-2 is that there is no
KV-cache here — each of the 784 passes re-runs the full image through the model.
"""

import os
import sys
import time
import warnings

import numpy as np
import torch
from typing import List, Tuple, Dict
from tqdm.auto import tqdm

# Reuse the arithmetic coder from the text experiment — no duplication
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, "..", "lm_arithmetic_coding"))
from arithmetic_coder import (
    ArithmeticEncoder, ArithmeticDecoder,
    bits_to_bytes, bytes_to_bits,
)

from pixelcnn import PixelCNN

SCALE_TOTAL    = 10_000_000
MIN_PROB       = 1e-9
CHECKPOINT_DIR = os.path.join(_HERE, "checkpoints")
IMG_H, IMG_W   = 28, 28
N_PIXELS       = IMG_H * IMG_W    # 784


# ---------------------------------------------------------------------------
# Singleton model loader
# ---------------------------------------------------------------------------

_model      = None
_model_path = None


def load_model(checkpoint_path: str = None) -> PixelCNN:
    global _model, _model_path
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "pixelcnn_fmnist.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint at '{checkpoint_path}'. Run train.py first."
        )
    if _model is None or _model_path != checkpoint_path:
        device = _get_device()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ckpt = torch.load(checkpoint_path, map_location=device)
        cfg    = ckpt.get("config", {"channels": 64, "blocks": 8})
        _model = PixelCNN(n_channels=cfg["channels"], n_residual_blocks=cfg["blocks"])
        _model.load_state_dict(ckpt["model"])
        _model.eval()
        _model.to(device)
        _model_path = checkpoint_path
        bpp = ckpt.get("test_bpp", "?")
        print(f"[pixel_compressor] Loaded PixelCNN "
              f"(channels={cfg['channels']}, blocks={cfg['blocks']}, "
              f"best test BPP={bpp})")
    return _model


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# CDF helper — identical logic to lm_compressor, same SCALE_TOTAL
# ---------------------------------------------------------------------------

def probs_to_int_cdf(probs: np.ndarray) -> Tuple[List[int], int]:
    """Convert a 256-element probability vector to an integer CDF."""
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

    cdf = [0] * 257          # 256 pixel values → 257-element CDF
    for i, c in enumerate(counts):
        cdf[i + 1] = cdf[i] + int(c)
    return cdf, total


# ---------------------------------------------------------------------------
# Theoretical BPP via cross-entropy (batched, no arithmetic coding)
# ---------------------------------------------------------------------------

def compute_bpp(
    images: np.ndarray,
    checkpoint_path: str = None,
    batch_size: int = 32,
) -> Tuple[float, Dict]:
    """
    Compute PixelCNN cross-entropy BPP on a batch of images.

    This is the theoretical compressed size — equivalent to compute_cross_entropy_bpc
    in the text experiment.  One forward pass per batch; very fast.

    Parameters
    ----------
    images : np.ndarray, shape (N, 28, 28), dtype uint8
    Returns
    -------
    (mean_bpp, metrics)
    """
    model    = load_model(checkpoint_path)
    device   = _get_device()
    n_images = len(images)
    n_pixels = n_images * N_PIXELS
    t_start  = time.time()
    total_nll = 0.0

    with tqdm(total=n_images, desc="Computing BPP", unit="img") as pbar:
        for i in range(0, n_images, batch_size):
            batch  = torch.from_numpy(
                images[i : i + batch_size]
            ).unsqueeze(1).to(device)           # (B, 1, 28, 28) int64
            target = batch.squeeze(1).long()    # (B, 28, 28)

            with torch.no_grad():
                logits = model(batch)           # (B, 256, 28, 28)

            nll = torch.nn.functional.cross_entropy(logits, target, reduction="sum")
            total_nll += nll.item()
            pbar.update(batch.shape[0])

    elapsed = time.time() - t_start
    bpp     = (total_nll / n_pixels) / np.log(2)   # nats → bits

    metrics = {
        "n_images":          n_images,
        "n_pixels":          n_pixels,
        "mean_bpp":          round(bpp, 6),
        "images_per_second": round(n_images / elapsed, 2),
        "elapsed_seconds":   round(elapsed, 2),
        "n_forward_passes":  int(np.ceil(n_images / batch_size)),
    }
    return bpp, metrics


# ---------------------------------------------------------------------------
# Full encode — 784 sequential forward passes (mirrors decoder exactly)
# ---------------------------------------------------------------------------

def pixel_encode(
    image: np.ndarray,
    checkpoint_path: str = None,
) -> Tuple[bytes, int, Dict]:
    """
    Losslessly encode a single 28×28 uint8 image using PixelCNN + arithmetic coding.

    WHY SEQUENTIAL (not single-pass):
    In exact arithmetic a single forward pass produces identical distributions
    to the decoder, because masked convolutions block future pixels.  In float32,
    however, tiny numerical differences accumulate across 8 residual blocks when the
    model input differs (full image during encode vs. partial image during decode).
    These sub-ULP differences shift the integer CDF by >=1 count, making the
    arithmetic coder encode and decode with mismatched ranges — losslessness fails.

    The fix: both encoder and decoder run 784 sequential passes, always feeding the
    same partial image (pixels known so far, rest 0).  Model inputs are byte-for-byte
    identical at every step, so CDF distributions match exactly.

    Speed note: encoding is now the same cost as decoding (~2-3 s per image on T4).
    The theoretical single-pass advantage holds in exact arithmetic but breaks in
    float32 with deep residual networks.

    Returns
    -------
    (compressed_bytes, n_bits, metrics)
    """
    assert image.shape == (IMG_H, IMG_W), f"Expected ({IMG_H},{IMG_W}) image, got {image.shape}"

    model   = load_model(checkpoint_path)
    device  = _get_device()
    t_start = time.time()

    # Context buffer — identical to what the decoder maintains
    context = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    enc     = ArithmeticEncoder()

    with tqdm(total=N_PIXELS, desc="Encoding pixels", unit="px",
              dynamic_ncols=True) as pbar:
        for idx in range(N_PIXELS):
            row, col = divmod(idx, IMG_W)

            # Feed the same partial image the decoder will use at this step
            img_t = torch.from_numpy(context).unsqueeze(0).unsqueeze(0).long().to(device)
            with torch.no_grad():
                logits = model(img_t)               # (1, 256, 28, 28)

            probs      = torch.softmax(logits[0, :, row, col], dim=0).cpu().numpy()
            cdf, total = probs_to_int_cdf(probs)
            pv         = int(image[row, col])
            enc.encode(cdf[pv], cdf[pv + 1], total)

            # Reveal true pixel so subsequent steps have correct context
            context[row, col] = pv
            pbar.update(1)

    bits       = enc.finish()
    compressed = bits_to_bytes(bits)
    elapsed    = time.time() - t_start

    metrics = {
        "n_pixels":         N_PIXELS,
        "n_forward_passes": N_PIXELS,          # 784 — same cost as decoder
        "compressed_bytes": len(compressed),
        "n_bits":           len(bits),
        "elapsed_seconds":  round(elapsed, 2),
    }
    return compressed, len(bits), metrics


# ---------------------------------------------------------------------------
# Full decode — 784 sequential forward passes
# ---------------------------------------------------------------------------

def pixel_decode(
    compressed: bytes,
    n_bits: int,
    checkpoint_path: str = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Decode bytes produced by pixel_encode back to a 28×28 uint8 image.

    Decoding is inherently sequential: to decode pixel i we need its distribution
    P(x_i | x_<i), which requires x_<i — so each pixel must be fully decoded
    before the next one can be predicted.  This requires 784 model forward passes
    per image, compared to just 1 during encoding.

    Returns
    -------
    (recovered_image: np.ndarray shape (28,28) uint8, metrics)
    """
    model  = load_model(checkpoint_path)
    device = _get_device()

    bits    = bytes_to_bits(compressed)[:n_bits]
    dec     = ArithmeticDecoder(bits)
    decoded = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    t_start = time.time()

    with tqdm(total=N_PIXELS, desc="Decoding pixels", unit="px",
              dynamic_ncols=True) as pbar:
        for idx in range(N_PIXELS):
            row, col = divmod(idx, IMG_W)

            # Feed partially-decoded image; unset pixels are 0 and are masked out
            img_t = torch.from_numpy(decoded).unsqueeze(0).unsqueeze(0).long().to(device)
            with torch.no_grad():
                logits = model(img_t)               # (1, 256, 28, 28)

            probs      = torch.softmax(logits[0, :, row, col], dim=0).cpu().numpy()
            cdf, total = probs_to_int_cdf(probs)
            pv         = dec.decode(cdf, total)
            decoded[row, col] = pv
            pbar.update(1)

    elapsed = time.time() - t_start
    metrics = {
        "n_pixels":          N_PIXELS,
        "n_forward_passes":  N_PIXELS,           # 784 — the decode cost
        "elapsed_seconds":   round(elapsed, 2),
        "pixels_per_second": round(N_PIXELS / elapsed, 1),
    }
    return decoded, metrics
