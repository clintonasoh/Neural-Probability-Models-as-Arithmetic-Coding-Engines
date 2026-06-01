"""
download_data.py
----------------
Downloads Enwik8 from mattmahoney.net and extracts a byte slice for experiments.

Usage:
    python download_data.py                          # 1MB from byte 0
    python download_data.py --size 500000            # 500KB
    python download_data.py --offset 5000000         # 1MB starting at byte 5M (clean prose)
"""

import os
import zipfile
import urllib.request
import argparse

DATA_DIR = "data"
URL      = "http://mattmahoney.net/dc/enwik8.zip"
ZIP_PATH = os.path.join(DATA_DIR, "enwik8.zip")
RAW_PATH = os.path.join(DATA_DIR, "enwik8")


def download_enwik8(size: int = 1_000_000, offset: int = 0) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    tag      = f"off{offset // 1000}k" if offset > 0 else "start"
    out_path = os.path.join(DATA_DIR, f"enwik8_{size // 1000}k_{tag}.txt")

    if os.path.exists(out_path):
        print(f"[download] {out_path} already exists, skipping download.")
        return out_path

    if not os.path.exists(ZIP_PATH):
        print(f"[download] Downloading Enwik8 from mattmahoney.net (~36MB)...")
        def progress(count, block_size, total_size):
            if total_size > 0:
                pct = min(count * block_size / total_size * 100, 100)
                print(f"  {pct:.1f}%", end="\r", flush=True)
        urllib.request.urlretrieve(URL, ZIP_PATH, reporthook=progress)
        print(f"\n[download] Saved to {ZIP_PATH}")
    else:
        print(f"[download] Using cached {ZIP_PATH}")

    if not os.path.exists(RAW_PATH):
        print(f"[download] Extracting zip...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extract("enwik8", DATA_DIR)
        print(f"[download] Extracted to {RAW_PATH}")

    print(f"[download] Slicing {size:,} bytes from offset {offset:,}...")
    with open(RAW_PATH, "rb") as f:
        f.seek(offset)
        data = f.read(size)

    with open(out_path, "wb") as f:
        f.write(data)

    print(f"[download] Saved {len(data):,} bytes to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size",   type=int, default=1_000_000)
    parser.add_argument("--offset", type=int, default=0,
                        help="Byte offset into enwik8 (default 0). "
                             "Use e.g. 5000000 to skip the XML header.")
    args = parser.parse_args()
    download_enwik8(args.size, args.offset)
