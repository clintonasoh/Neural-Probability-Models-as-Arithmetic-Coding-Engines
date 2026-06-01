"""
evaluate.py
-----------
Main evaluation script — BLM6106 Data Compression Project, Approach 1.

Tracks compression ratio, encoding speed, and LLM token usage.
All results are saved to the results/ folder as JSON + a formatted text report.

Usage:
    python evaluate.py                            # gpt2, 1MB from start
    python evaluate.py --model gpt2-medium        # larger model
    python evaluate.py --offset 5000000           # skip XML header (cleaner prose)
    python evaluate.py --size 100000              # smaller slice for quick tests
    python evaluate.py --full-encode              # also run actual encode/decode
"""

import os
import sys
import json
import time
import math
import psutil
import argparse
import warnings
import datetime
warnings.filterwarnings("ignore", message=".*loss_type=None.*")
warnings.filterwarnings("ignore", message=".*sequence length.*")

from download_data  import download_enwik8
from baselines      import run_baselines
from lm_compressor  import compute_cross_entropy_bpc, lm_encode, lm_decode

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def make_run_id(model: str, size: int, offset: int) -> str:
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"off{offset//1000}k" if offset > 0 else "start"
    return f"{model}_{size//1000}k_{tag}_{ts}"


def print_table(results: dict, metrics: dict, original_bytes: int):
    raw_bpc = 8.0
    print()
    print("=" * 78)
    print(f"  {'Method':<38} {'BPC':>6}  {'Ratio':>6}  {'Saving':>7}  {'Speed':>12}")
    print("-" * 78)
    for name, bpc in sorted(results.items(), key=lambda x: x[1]):
        ratio  = raw_bpc / bpc
        saving = (1 - bpc / raw_bpc) * 100
        spd    = metrics.get(name, {}).get("speed_str", "—")
        print(f"  {name:<38} {bpc:>6.3f}  {ratio:>5.2f}x  {saving:>6.1f}%  {spd:>12}")
    print("=" * 78)
    print(f"  Original size : {original_bytes:,} bytes  ({original_bytes/1024:.1f} KB)")
    print()


def save_results(run_id: str, results: dict, metrics: dict,
                 original_bytes: int, args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw_bpc = 8.0

    # JSON
    payload = {
        "run_id":         run_id,
        "timestamp":      datetime.datetime.now().isoformat(),
        "config": {
            "model":   args.model,
            "size":    args.size,
            "offset":  args.offset,
            "window":  args.window,
        },
        "original_bytes": original_bytes,
        "results": {
            name: {
                "bpc":            round(bpc, 6),
                "compression_ratio": round(raw_bpc / bpc, 4),
                "space_saving_pct":  round((1 - bpc / raw_bpc) * 100, 2),
                **metrics.get(name, {}),
            }
            for name, bpc in results.items()
        }
    }
    json_path = os.path.join(RESULTS_DIR, f"{run_id}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Human-readable report
    txt_path = os.path.join(RESULTS_DIR, f"{run_id}_report.txt")
    with open(txt_path, "w") as f:
        f.write("BLM6106 Data Compression — LLM Arithmetic Coding Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Run ID   : {run_id}\n")
        f.write(f"Date     : {payload['timestamp']}\n")
        f.write(f"Model    : {args.model}\n")
        f.write(f"Dataset  : Enwik8, {args.size//1000}KB from offset {args.offset:,}\n")
        f.write(f"Window   : {args.window} tokens\n\n")
        f.write(f"{'Method':<40} {'BPC':>6}  {'Ratio':>6}  {'Saving':>7}\n")
        f.write("-" * 60 + "\n")
        for name, bpc in sorted(results.items(), key=lambda x: x[1]):
            ratio  = raw_bpc / bpc
            saving = (1 - bpc / raw_bpc) * 100
            f.write(f"{name:<40} {bpc:>6.3f}  {ratio:>5.2f}x  {saving:>6.1f}%\n")
        f.write("\nDetailed Metrics\n" + "-" * 60 + "\n")
        for name, m in metrics.items():
            if m:
                f.write(f"\n[{name}]\n")
                for k, v in m.items():
                    if k != "speed_str":
                        f.write(f"  {k}: {v}\n")

    print(f"[results] Saved → {json_path}")
    print(f"[results] Saved → {txt_path}")
    return json_path, txt_path


# ---------------------------------------------------------------------------
# Memory helper
# ---------------------------------------------------------------------------

def current_memory_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM Arithmetic Coding — BLM6106 Evaluation"
    )
    parser.add_argument("--size",        type=int,  default=1_000_000)
    parser.add_argument("--offset",      type=int,  default=0,
                        help="Byte offset into enwik8 (use 5000000 for prose)")
    parser.add_argument("--model",       type=str,  default="gpt2")
    parser.add_argument("--window",      type=int,  default=512)
    parser.add_argument("--full-encode", action="store_true",
                        help="Run actual LM encode/decode on a 5KB sample")
    args = parser.parse_args()

    run_id  = make_run_id(args.model, args.size, args.offset)
    results = {}
    metrics = {}

    # ---- 1. Data -----------------------------------------------------------
    print(f"\n[1/4] Preparing Enwik8 (size={args.size//1000}KB, offset={args.offset:,})...")
    data_path = download_enwik8(args.size, args.offset)
    with open(data_path, "rb") as f:
        text = f.read()
    print(f"      Loaded {len(text):,} bytes")

    # ---- 2. Baselines ------------------------------------------------------
    print("\n[2/4] Running baseline compressors...")
    mem_before = current_memory_mb()
    t0         = time.time()
    baselines  = run_baselines(text)
    bl_time    = time.time() - t0
    results.update(baselines)

    throughput = len(text) / bl_time / 1024
    for name in baselines:
        metrics[name] = {
            "elapsed_seconds":   round(bl_time, 2),
            "throughput_kb_s":   round(throughput, 1),
            "speed_str":         f"{throughput:.0f} KB/s",
        }
    print(f"      Done in {bl_time:.1f}s")
    for name, bpc in baselines.items():
        print(f"      {name}: {bpc:.4f} bpc")

    # ---- 3. LLM cross-entropy (theoretical BPC) ----------------------------
    print(f"\n[3/4] Computing {args.model} cross-entropy BPC (with KV-cache)...")
    mem_before_lm = current_memory_mb()
    bpt, bpc, lm_metrics = compute_cross_entropy_bpc(
        text, window=args.window, model_name=args.model
    )
    mem_after_lm = current_memory_mb()
    lm_metrics["peak_memory_delta_mb"] = round(mem_after_lm - mem_before_lm, 1)

    label = f"LLM-AC ({args.model}, theoretical)"
    results[label]                     = bpc
    lm_metrics["speed_str"]            = f"{lm_metrics['tokens_per_second']:.1f} tok/s"
    metrics[label]                     = lm_metrics

    print(f"      {bpt:.4f} bits/token  →  {bpc:.4f} bits/char")
    print(f"      Speed: {lm_metrics['tokens_per_second']:.1f} tok/s  |  "
          f"Forward passes: {lm_metrics['n_forward_passes']:,}  |  "
          f"Mem Δ: {lm_metrics['peak_memory_delta_mb']:.0f} MB")

    # ---- 4. Actual encode/decode (optional) --------------------------------
    if args.full_encode:
        sample_size = 5_000
        print(f"\n[4/4] Full LM encode/decode on first {sample_size} bytes...")
        sample = text[:sample_size]

        compressed, token_ids, n_bits, enc_metrics = lm_encode(
            sample, window=args.window, model_name=args.model
        )
        recovered, dec_metrics = lm_decode(
            compressed, token_ids, n_bits,
            window=args.window, model_name=args.model
        )

        original_str  = sample.decode("utf-8", errors="replace")
        recovered_str = recovered.decode("utf-8", errors="replace")
        lossless      = (original_str == recovered_str)
        actual_bpc    = n_bits / len(sample)

        enc_label = f"LLM-AC ({args.model}, actual {sample_size//1000}KB)"
        results[enc_label] = actual_bpc
        metrics[enc_label] = {
            **enc_metrics,
            "lossless":         lossless,
            "decode_time_s":    dec_metrics["elapsed_seconds"],
            "decode_tok_s":     dec_metrics["tokens_per_second"],
            "speed_str":        f"{enc_metrics['tokens_per_second']:.1f} tok/s",
        }

        status = "✓ LOSSLESS" if lossless else "✗ NOT LOSSLESS"
        print(f"\n  {status}")
        print(f"  Original: {len(sample):,} bytes → Compressed: {len(compressed):,} bytes")
        print(f"  Actual BPC : {actual_bpc:.4f}")
        print(f"  Encode     : {enc_metrics['elapsed_seconds']:.1f}s  "
              f"({enc_metrics['tokens_per_second']:.1f} tok/s)")
        print(f"  Decode     : {dec_metrics['elapsed_seconds']:.1f}s  "
              f"({dec_metrics['tokens_per_second']:.1f} tok/s)")
        print(f"  LLM passes : encode={enc_metrics['n_forward_passes']}  "
              f"decode={dec_metrics['n_forward_passes']}")

        if not lossless:
            for i, (a, b) in enumerate(zip(original_str, recovered_str)):
                if a != b:
                    print(f"  First mismatch at char {i}: {repr(a)} vs {repr(b)}")
                    break
    else:
        print("\n[4/4] Skipped full encode/decode (use --full-encode to enable).")

    # ---- Summary -----------------------------------------------------------
    print_table(results, metrics, len(text))
    save_results(run_id, results, metrics, len(text), args)


if __name__ == "__main__":
    main()
