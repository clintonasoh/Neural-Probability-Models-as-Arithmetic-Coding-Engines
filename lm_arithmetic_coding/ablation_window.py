"""
ablation_window.py
------------------
Window size ablation for LLM-AC on Enwik8.

Iterates over window sizes [128, 256, 512], reusing any existing result
that matches (model, size, window) rather than re-running it, then prints
a consolidated comparison table and saves an ablation summary JSON.

Usage:
    python ablation_window.py                         # gpt2, 50KB, windows 128/256/512
    python ablation_window.py --windows 64 128 256 512
    python ablation_window.py --model gpt2-medium
    python ablation_window.py --force                 # ignore cached results, re-run all
"""

import os
import sys
import json
import glob
import argparse
import subprocess
from datetime import datetime

_HERE       = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")
EVALUATE    = os.path.join(_HERE, "evaluate.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_existing(model: str, size: int, window: int, offset: int = 0):
    """
    Search results/ for a JSON whose config matches (model, size, window, offset).
    Returns (path, data) or (None, None).
    """
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
            cfg = data.get("config", {})
            if (cfg.get("model")        == model  and
                cfg.get("size")         == size   and
                cfg.get("window")       == window and
                cfg.get("offset", 0)    == offset):
                return path, data
        except Exception:
            continue
    return None, None


def run_evaluate(model: str, size: int, window: int, offset: int = 0):
    """
    Invoke evaluate.py as a subprocess and return (path, data) for the
    freshest JSON written to results/ afterwards.
    """
    cmd = [
        sys.executable, EVALUATE,
        "--model",  model,
        "--size",   str(size),
        "--window", str(window),
        "--offset", str(offset),
    ]
    print(f"\n[ablation] Running: {' '.join(cmd[1:])}")
    subprocess.run(cmd, cwd=_HERE)

    # Pick the most-recently modified JSON that matches the model prefix
    candidates = sorted(
        glob.glob(os.path.join(RESULTS_DIR, f"{model.replace('/', '_')}*.json")),
        key=os.path.getmtime,
    )
    if candidates:
        with open(candidates[-1]) as f:
            return candidates[-1], json.load(f)
    return None, None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(records: list, model: str, size: int) -> None:
    RAW_BPC  = 8.0
    bpc_512  = next((r["bpc"] for r in records if r["window"] == 512), None)

    print()
    print("=" * 76)
    print(f"  Window Size Ablation — {model}, {size//1000}KB, Enwik8")
    print("=" * 76)
    print(f"  {'Window':>8}  {'BPC':>8}  {'Ratio':>7}  {'Saving':>8}  {'vs w=512':>10}  {'Speed':>10}")
    print("-" * 76)

    for r in sorted(records, key=lambda x: x["window"]):
        ratio  = RAW_BPC / r["bpc"]
        saving = (1 - r["bpc"] / RAW_BPC) * 100
        if bpc_512 is None or r["window"] == 512:
            delta = "baseline"
        else:
            delta = f"{r['bpc'] - bpc_512:+.4f}"
        speed = f"{r['toks']:.1f} tok/s" if r["toks"] else "—"
        print(f"  {r['window']:>8}  {r['bpc']:>8.4f}  {ratio:>6.2f}x  {saving:>7.1f}%  "
              f"{delta:>10}  {speed:>10}")

    print("=" * 76)
    print()
    if bpc_512:
        if len(records) > 1:
            worst = max(r["bpc"] for r in records)
            print(f"  Context impact : {worst - bpc_512:+.4f} BPC from shortest to longest window")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Window size ablation for LLM-AC"
    )
    parser.add_argument("--windows", type=int, nargs="+", default=[128, 256, 512],
                        help="Window sizes to evaluate (default: 128 256 512)")
    parser.add_argument("--model",   type=str, default="gpt2",
                        help="HuggingFace model name (default: gpt2)")
    parser.add_argument("--size",    type=int, default=50_000,
                        help="Bytes of Enwik8 to use (default: 50000)")
    parser.add_argument("--offset",  type=int, default=0,
                        help="Byte offset into enwik8 (default: 0; use 300000 for prose)")
    parser.add_argument("--force",   action="store_true",
                        help="Re-run even when a cached result exists")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    records = []

    for window in sorted(args.windows):
        print(f"\n{'='*60}")
        print(f"  Window = {window} tokens  |  model = {args.model}  |  size = {args.size//1000}KB")
        print(f"{'='*60}")

        path, data = None, None

        if not args.force:
            path, data = find_existing(args.model, args.size, window, args.offset)
            if data:
                print(f"  [ablation] Reusing cached result: {os.path.basename(path)}")

        if data is None:
            path, data = run_evaluate(args.model, args.size, window, args.offset)

        if data is None:
            print(f"  [ablation] ERROR: could not obtain result for window={window}. Skipping.")
            continue

        llm_key = next((k for k in data["results"] if "theoretical" in k), None)
        if llm_key is None:
            print(f"  [ablation] No theoretical BPC key found in {path}. Skipping.")
            continue

        info = data["results"][llm_key]
        bpc  = info["bpc"]
        toks = info.get("tokens_per_second", 0)
        records.append({"window": window, "bpc": bpc, "toks": toks, "path": path})
        print(f"  BPC = {bpc:.4f}  |  {toks:.1f} tok/s")

    if not records:
        print("\n[ablation] No results collected — nothing to report.")
        return

    print_table(records, args.model, args.size)

    # Save summary JSON
    ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(
        RESULTS_DIR,
        f"ablation_window_{args.model.replace('/', '_')}_{args.size//1000}k_off{args.offset//1000}k_w{'_'.join(str(w) for w in sorted(args.windows))}_{ts}.json"
    )
    with open(summary_path, "w") as f:
        json.dump({
            "ablation":  "window_size",
            "model":     args.model,
            "size":      args.size,
            "offset":    args.offset,
            "timestamp": datetime.now().isoformat(),
            "windows_tested": sorted(args.windows),
            "results": [
                {
                    "window":           r["window"],
                    "bpc":              round(r["bpc"], 6),
                    "tokens_per_second": round(r["toks"], 2),
                    "source_json":      os.path.basename(r["path"]),
                }
                for r in sorted(records, key=lambda x: x["window"])
            ],
        }, f, indent=2)

    print(f"[ablation] Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
