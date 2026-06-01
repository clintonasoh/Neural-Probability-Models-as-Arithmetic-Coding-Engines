"""
train.py
--------
Train PixelCNN on Fashion MNIST and save the best checkpoint.

Usage:
    python train.py                        # 20 epochs, 64 channels, 8 residual blocks
    python train.py --epochs 30            # more epochs
    python train.py --channels 128         # larger model (~2× VRAM, ~0.15 BPP improvement)
    python train.py --resume               # continue from last checkpoint

Expected runtime on Tesla T4 (14.6 GB VRAM):
    64 channels,  8 blocks, 20 epochs → ~12–15 minutes, ~3.7 BPP
    128 channels, 8 blocks, 20 epochs → ~25–30 minutes, ~3.5 BPP

Checkpoint saved to: checkpoints/pixelcnn_fmnist.pt
Training log saved to: results/training_log.json
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from pixelcnn import PixelCNN
from download_data import load_fashion_mnist, download_fashion_mnist

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
RESULTS_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
NATS_TO_BITS   = 1.0 / torch.tensor(2.0).log().item()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(images: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Wrap a uint8 numpy array as a DataLoader of int64 tensors."""
    t  = torch.from_numpy(images).unsqueeze(1).long()   # (N, 1, 28, 28)
    ds = TensorDataset(t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, pin_memory=True)


def run_epoch(model, loader, optimizer, device, train: bool) -> float:
    """Run one train or eval epoch; return mean NLL in nats/pixel."""
    model.train(train)
    total_nll    = 0.0
    total_pixels = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for (batch,) in loader:
            batch  = batch.to(device)                       # (B, 1, 28, 28) int64
            logits = model(batch)                           # (B, 256, 28, 28)
            target = batch.squeeze(1)                       # (B, 28, 28) int64
            loss   = nn.functional.cross_entropy(logits, target)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            n_px          = batch.size(0) * 28 * 28
            total_nll    += loss.item() * n_px
            total_pixels += n_px

    return total_nll / total_pixels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train PixelCNN on Fashion MNIST")
    parser.add_argument("--epochs",   type=int,   default=20)
    parser.add_argument("--channels", type=int,   default=64,
                        help="Feature channels (64=fast/~3.7BPP, 128=better/~3.5BPP)")
    parser.add_argument("--blocks",   type=int,   default=8,
                        help="Number of residual blocks")
    parser.add_argument("--batch",    type=int,   default=128)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--resume",   action="store_true",
                        help="Resume from existing checkpoint")
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()
    print(f"[train] Device : {device}")
    if device.type == "cuda":
        print(f"[train] GPU    : {torch.cuda.get_device_name(0)}")
        print(f"[train] VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Data --------------------------------------------------------------
    try:
        train_images, _ = load_fashion_mnist("train")
        test_images,  _ = load_fashion_mnist("test")
    except FileNotFoundError:
        download_fashion_mnist()
        train_images, _ = load_fashion_mnist("train")
        test_images,  _ = load_fashion_mnist("test")

    train_loader = make_loader(train_images, args.batch, shuffle=True)
    test_loader  = make_loader(test_images,  args.batch, shuffle=False)
    print(f"[train] Train : {len(train_images):,} images")
    print(f"[train] Test  : {len(test_images):,} images")

    # ---- Model -------------------------------------------------------------
    model    = PixelCNN(n_channels=args.channels, n_residual_blocks=args.blocks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model : PixelCNN  channels={args.channels}  blocks={args.blocks}")
    print(f"[train] Params: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    ckpt_path   = os.path.join(CHECKPOINT_DIR, "pixelcnn_fmnist.pt")
    start_epoch = 0

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[train] Resumed from epoch {start_epoch}  "
              f"(best test BPP was {ckpt['test_bpp']:.4f})")

    # ---- Training loop -----------------------------------------------------
    print(f"\n{'Epoch':>6}  {'Train BPP':>10}  {'Test BPP':>9}  {'Time':>7}  {'Note'}")
    print("-" * 60)

    best_test_bpp = float("inf")
    history       = []

    for epoch in range(start_epoch, args.epochs):
        t0        = time.time()
        train_nll = run_epoch(model, train_loader, optimizer, device, train=True)
        test_nll  = run_epoch(model, test_loader,  optimizer, device, train=False)
        scheduler.step()

        train_bpp = train_nll * NATS_TO_BITS
        test_bpp  = test_nll  * NATS_TO_BITS
        elapsed   = time.time() - t0
        note      = ""

        if test_bpp < best_test_bpp:
            best_test_bpp = test_bpp
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "test_bpp":  round(test_bpp, 6),
                "config": {
                    "channels": args.channels,
                    "blocks":   args.blocks,
                },
            }, ckpt_path)
            note = "✓ saved"

        print(f"{epoch+1:>6}  {train_bpp:>10.4f}  {test_bpp:>9.4f}  {elapsed:>6.1f}s  {note}")
        history.append({
            "epoch": epoch + 1,
            "train_bpp": round(train_bpp, 6),
            "test_bpp":  round(test_bpp,  6),
            "elapsed_s": round(elapsed, 1),
        })

    # ---- Save training log -------------------------------------------------
    log_path = os.path.join(RESULTS_DIR, "training_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "config": vars(args),
            "best_test_bpp": round(best_test_bpp, 6),
            "history": history,
        }, f, indent=2)

    print(f"\n[train] Best test BPP : {best_test_bpp:.4f}")
    print(f"[train] Checkpoint    : {ckpt_path}")
    print(f"[train] Training log  : {log_path}")


if __name__ == "__main__":
    main()
