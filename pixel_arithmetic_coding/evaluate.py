"""
evaluate.py
-----------
Main evaluation script — Fashion MNIST PixelCNN arithmetic coding.
Analog of ../lm_arithmetic_coding/evaluate.py.

Usage:
    python evaluate.py                           # 100 images, theoretical BPP only
    python evaluate.py --n-images 500            # larger sample for stable estimates
    python evaluate.py --full-encode             # also run actual encode/decode (5 images)
    python evaluate.py --full-encode --n-encode 20
    python evaluate.py --checkpoint path/to.pt  # custom checkpoint

Saves timestamped JSON + human-readable report to results/.
"""

import os
import sys
import json
import time
import argparse
import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from download_data    import load_fashion_mnist, download_fashion_mnist, CLASS_NAMES
from baselines        import run_baselines_batch
from pixel_compressor import compute_bpp, pixel_encode, pixel_decode

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RAW_BPP     = 8.0
IMG_H, IMG_W = 28, 28
N_PIXELS     = IMG_H * IMG_W   # 784


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_run_id(n_images: int) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"pixelcnn_fmnist_{n_images}img_{ts}"


def print_table(results: dict) -> None:
    print()
    print("=" * 74)
    print(f"  {'Method':<38} {'BPP':>6}  {'Ratio':>6}  {'Saving':>7}")
    print("-" * 74)
    for name, bpp in sorted(results.items(), key=lambda x: x[1]):
        ratio  = RAW_BPP / bpp
        saving = (1 - bpp / RAW_BPP) * 100
        print(f"  {name:<38} {bpp:>6.3f}  {ratio:>5.2f}x  {saving:>6.1f}%")
    print("=" * 74)
    print()


def save_results(run_id: str, results: dict, metrics: dict, args) -> tuple:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    payload = {
        "run_id":    run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "n_images":    args.n_images,
            "checkpoint":  args.checkpoint,
            "full_encode": args.full_encode,
            "n_encode":    args.n_encode,
        },
        "results": {
            name: {
                "bpp":               round(bpp, 6),
                "compression_ratio": round(RAW_BPP / bpp, 4),
                "space_saving_pct":  round((1 - bpp / RAW_BPP) * 100, 2),
                **metrics.get(name, {}),
            }
            for name, bpp in results.items()
        },
    }

    json_path = os.path.join(RESULTS_DIR, f"{run_id}.json")
    txt_path  = os.path.join(RESULTS_DIR, f"{run_id}_report.txt")

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    with open(txt_path, "w") as f:
        f.write("BLM6106 — PixelCNN Arithmetic Coding (Fashion MNIST)\n")
        f.write("=" * 65 + "\n")
        f.write(f"Run     : {run_id}\n")
        f.write(f"Date    : {payload['timestamp']}\n")
        f.write(f"Images  : {args.n_images} from Fashion MNIST test split\n\n")
        f.write(f"{'Method':<38} {'BPP':>6}  {'Ratio':>6}  {'Saving':>7}\n")
        f.write("-" * 60 + "\n")
        for name, bpp in sorted(results.items(), key=lambda x: x[1]):
            ratio  = RAW_BPP / bpp
            saving = (1 - bpp / RAW_BPP) * 100
            f.write(f"{name:<38} {bpp:>6.3f}  {ratio:>5.2f}x  {saving:>6.1f}%\n")
        f.write("\nDetailed Metrics\n" + "-" * 60 + "\n")
        for name, m in metrics.items():
            if m:
                f.write(f"\n[{name}]\n")
                for k, v in m.items():
                    f.write(f"  {k}: {v}\n")

    print(f"[results] Saved → {json_path}")
    print(f"[results] Saved → {txt_path}")
    return json_path, txt_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PixelCNN Arithmetic Coding — Fashion MNIST Evaluation"
    )
    parser.add_argument("--n-images",    type=int,  default=100,
                        help="Number of test images for BPP evaluation (default 100)")
    parser.add_argument("--full-encode", action="store_true",
                        help="Run actual arithmetic encode+decode on --n-encode images")
    parser.add_argument("--n-encode",    type=int,  default=5,
                        help="Images to fully encode/decode (default 5)")
    parser.add_argument("--checkpoint",  type=str,  default=None,
                        help="Path to PixelCNN checkpoint")
    args = parser.parse_args()

    run_id  = make_run_id(args.n_images)
    results = {}
    metrics = {}

    # ---- 1. Data -----------------------------------------------------------
    print(f"\n[1/4] Loading Fashion MNIST test split...")
    try:
        images, labels = load_fashion_mnist("test")
    except FileNotFoundError:
        download_fashion_mnist()
        images, labels = load_fashion_mnist("test")

    sample   = images[:args.n_images]
    slabels  = labels[:args.n_images]
    classes  = sorted(set(slabels.tolist()))
    print(f"      {len(sample)} images  |  "
          f"{len(classes)}/10 classes  |  "
          f"pixel range [{sample.min()}–{sample.max()}]")

    # ---- 2. Baselines ------------------------------------------------------
    print(f"\n[2/4] Running image baselines on {args.n_images} images...")
    t0 = time.time()
    bl = run_baselines_batch(sample)
    results.update(bl)
    for name, bpp in bl.items():
        metrics[name] = {"elapsed_seconds": round(time.time() - t0, 2)}
        print(f"      {name}: {bpp:.4f} bpp")

    # ---- 3. PixelCNN theoretical BPP (cross-entropy) -----------------------
    print(f"\n[3/4] Computing PixelCNN cross-entropy BPP ({args.n_images} images)...")
    bpp, lm_m = compute_bpp(sample, checkpoint_path=args.checkpoint)
    label = "PixelCNN-AC (theoretical)"
    results[label] = bpp
    metrics[label] = lm_m
    print(f"      {bpp:.4f} bpp  |  "
          f"{lm_m['images_per_second']:.1f} img/s  |  "
          f"{lm_m['n_forward_passes']} forward passes total")

    # ---- 4. Full encode/decode (optional) ----------------------------------
    if args.full_encode:
        n = args.n_encode
        print(f"\n[4/4] Full encode/decode on {n} images...")
        enc_bpps, lossless_count = [], 0
        per_image = []

        for i in range(n):
            img        = sample[i]
            cls_name   = CLASS_NAMES[slabels[i]]
            compressed, n_bits, enc_m = pixel_encode(img, checkpoint_path=args.checkpoint)
            recovered,           dec_m = pixel_decode(compressed, n_bits,
                                                       checkpoint_path=args.checkpoint)
            lossless = np.array_equal(img, recovered)
            bpp_i    = n_bits / (IMG_H * IMG_W)
            enc_bpps.append(bpp_i)
            if lossless:
                lossless_count += 1

            status = "✓ lossless" if lossless else "✗ MISMATCH"
            print(f"  [{i+1}/{n}] {cls_name:<14}  {status}  "
                  f"{bpp_i:.4f} bpp  "
                  f"enc={enc_m['elapsed_seconds']:.1f}s ({enc_m['n_forward_passes']} passes)  "
                  f"dec={dec_m['elapsed_seconds']:.1f}s (784 passes)")
            per_image.append({
                "class":       cls_name,
                "lossless":    lossless,
                "bpp":         round(bpp_i, 6),
                "enc_seconds": enc_m["elapsed_seconds"],
                "dec_seconds": dec_m["elapsed_seconds"],
            })

        mean_bpp  = float(np.mean(enc_bpps))
        enc_label = f"PixelCNN-AC (actual, {n} images)"
        results[enc_label] = mean_bpp
        metrics[enc_label] = {
            "lossless_count":  lossless_count,
            "lossless_pct":    round(lossless_count / n * 100, 1),
            "mean_actual_bpp": round(mean_bpp, 6),
            "per_image":       per_image,
        }
        print(f"\n  Lossless: {lossless_count}/{n}  |  "
              f"Mean actual BPP: {mean_bpp:.4f}")

        if lossless_count < n:
            print("  WARNING: not all images decoded losslessly — "
                  "check arithmetic coder or CDF scaling.")
    else:
        print("\n[4/4] Skipped full encode/decode (use --full-encode to enable).")

    # ---- Summary -----------------------------------------------------------
    print_table(results)
    save_results(run_id, results, metrics, args)


if __name__ == "__main__":
    main()
