"""
train.py — Training & Benchmarking Script
==========================================
Trains both standard U-Net and Half U-Net on a medical segmentation dataset
and logs the following metrics:

  Segmentation quality : Dice Similarity Coefficient (DSC)
  Efficiency metrics   : Parameter count, Peak GPU VRAM (MB), Inference latency (ms)

Usage
-----
  python train.py --data_dir ./data --dataset busi --epochs 20 --batch_size 8

The script saves:
  • Best model checkpoints  → checkpoints/{model}_best.pth
  • Training curves plot    → results/training_curves.png
  • Benchmark table         → results/benchmark_results.csv
"""

import os
import time
import csv
import argparse
import random

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import UNet, HalfUNet
from losses import BCEDiceLoss, dice_coefficient
from dataset import get_dataloaders


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_vram_mb(model: torch.nn.Module,
                    device: torch.device,
                    img_size: int = 256) -> float:
    """Run one forward pass and measure peak GPU VRAM allocation in MB."""
    if device.type != "cuda":
        return 0.0
    torch.cuda.reset_peak_memory_stats(device)
    dummy = torch.randn(1, 1, img_size, img_size, device=device)
    with torch.no_grad():
        _ = model(dummy)
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return peak_bytes / (1024 ** 2)


def measure_latency_ms(model: torch.nn.Module,
                       device: torch.device,
                       img_size: int = 256,
                       n_runs: int = 50) -> float:
    """Average inference latency over n_runs forward passes (ms)."""
    dummy = torch.randn(1, 1, img_size, img_size, device=device)
    # Warm-up
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_dsc = 0.0, 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_dsc  += dice_coefficient(logits.detach(), masks)
    n = len(loader)
    return total_loss / n, total_dsc / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_dsc = 0.0, 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        total_loss += criterion(logits, masks).item()
        total_dsc  += dice_coefficient(logits, masks)
    n = len(loader)
    return total_loss / n, total_dsc / n


# ─────────────────────────────────────────────
# Full Benchmark
# ─────────────────────────────────────────────

def run_benchmark(model_cls, model_name, args, device, train_loader, val_loader):
    print(f"\n{'='*55}")
    print(f"  Training {model_name}")
    print(f"{'='*55}")

    model     = model_cls(n_channels=1, n_classes=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs("checkpoints", exist_ok=True)
    best_dsc    = 0.0
    train_dscs, val_dscs = [], []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_dsc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_dsc = validate(model, val_loader, criterion, device)
        scheduler.step()

        train_dscs.append(tr_dsc)
        val_dscs.append(vl_dsc)

        if vl_dsc > best_dsc:
            best_dsc = vl_dsc
            torch.save(model.state_dict(),
                       f"checkpoints/{model_name}_best.pth")

        print(f"  Epoch {epoch:03d}/{args.epochs} | "
              f"Train Loss {tr_loss:.4f}  DSC {tr_dsc:.4f} | "
              f"Val Loss {vl_loss:.4f}  DSC {vl_dsc:.4f}")

    # ── Efficiency metrics ──────────────────────────────
    n_params     = count_parameters(model)
    peak_vram_mb = measure_vram_mb(model, device, args.img_size)
    latency_ms   = measure_latency_ms(model, device, args.img_size)

    print(f"\n  [{model_name}] Best Val DSC : {best_dsc:.4f}")
    print(f"  [{model_name}] Parameters   : {n_params:,}")
    print(f"  [{model_name}] Peak VRAM    : {peak_vram_mb:.1f} MB")
    print(f"  [{model_name}] Latency      : {latency_ms:.2f} ms")

    return {
        "model"       : model_name,
        "best_val_dsc": round(best_dsc,    4),
        "n_params"    : n_params,
        "peak_vram_mb": round(peak_vram_mb, 1),
        "latency_ms"  : round(latency_ms,   2),
        "train_dscs"  : train_dscs,
        "val_dscs"    : val_dscs,
    }


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_curves(results, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(results[0]["val_dscs"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"UNet": "#2563EB", "HalfUNet": "#DC2626"}

    for r in results:
        name = r["model"]
        c    = colors.get(name, "grey")
        axes[0].plot(epochs, r["train_dscs"], label=f"{name} – Train",
                     color=c, linewidth=2)
        axes[0].plot(epochs, r["val_dscs"],   label=f"{name} – Val",
                     color=c, linewidth=2, linestyle="--")

    axes[0].set_title("Dice Similarity Coefficient over Epochs", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("DSC")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Bar chart: param count comparison
    names    = [r["model"] for r in results]
    params_m = [r["n_params"] / 1e6 for r in results]
    bar_cols = [colors.get(n, "grey") for n in names]
    axes[1].bar(names, params_m, color=bar_cols, width=0.4, edgecolor="black", linewidth=0.8)
    axes[1].set_title("Trainable Parameters (Millions)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Parameters (M)")
    for i, v in enumerate(params_m):
        axes[1].text(i, v + 0.05, f"{v:.2f}M", ha="center", fontsize=11, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n[Plot] Saved to {path}")
    plt.close()


def save_csv(results, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "benchmark_results.csv")
    fieldnames = ["model", "best_val_dsc", "n_params", "peak_vram_mb", "latency_ms"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"[CSV]  Saved to {path}")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="U-Net vs Half U-Net Benchmark")
    parser.add_argument("--data_dir",   type=str,   required=True,
                        help="Root directory of the dataset")
    parser.add_argument("--dataset",    type=str,   default="busi",
                        choices=["busi", "isic"],
                        help="Dataset type: 'busi' or 'isic'")
    parser.add_argument("--img_size",   type=int,   default=256)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--num_workers",type=int,   default=2)
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    train_loader, val_loader = get_dataloaders(
        data_dir     = args.data_dir,
        dataset_type = args.dataset,
        img_size     = args.img_size,
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
    )

    results = []
    for model_cls, name in [(UNet, "UNet"), (HalfUNet, "HalfUNet")]:
        r = run_benchmark(model_cls, name, args, device, train_loader, val_loader)
        results.append(r)

    # ── Summary table ────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  {'Model':<14} {'DSC':>8} {'Params':>12} {'VRAM (MB)':>12} {'Latency (ms)':>14}")
    print(f"{'─'*65}")
    for r in results:
        print(f"  {r['model']:<14} {r['best_val_dsc']:>8.4f} "
              f"{r['n_params']:>12,} {r['peak_vram_mb']:>12.1f} {r['latency_ms']:>14.2f}")
    print(f"{'─'*65}")

    # Efficiency deltas
    if len(results) == 2:
        u, h = results[0], results[1]
        param_red = (1 - h["n_params"]  / u["n_params"])  * 100
        vram_red  = (1 - h["peak_vram_mb"] / u["peak_vram_mb"]) * 100 if u["peak_vram_mb"] else 0
        dsc_delta = (h["best_val_dsc"] - u["best_val_dsc"]) * 100
        print(f"\n  Parameter reduction : {param_red:.1f}%")
        print(f"  VRAM reduction      : {vram_red:.1f}%")
        print(f"  DSC delta           : {dsc_delta:+.2f} pp")

    plot_curves(results)
    save_csv(results)


if __name__ == "__main__":
    main()
