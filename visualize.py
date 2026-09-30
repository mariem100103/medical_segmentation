"""
visualize.py — Segmentation Inference & Visual Comparison
==========================================================
Loads the best saved checkpoints for U-Net and Half U-Net and produces
a publication-quality figure grid:

    Row layout per sample:
    [ Input Image | Ground Truth | U-Net Prediction | Half U-Net Prediction ]

Usage
-----
    python visualize.py \
        --data_dir  ./data \
        --dataset   busi \
        --n_samples 6 \
        --output    results/segmentation_comparison.png

Outputs
-------
    results/segmentation_comparison.png   — grid figure
    results/overlay_unet.png             — overlay (image + UNet mask)
    results/overlay_halfunet.png         — overlay (image + HalfUNet mask)
"""

import os
import random
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from models  import UNet, HalfUNet
from losses  import dice_coefficient
from dataset import BUSIDataset, ISICDataset

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────

def load_model(model_cls, checkpoint_path: str, device: torch.device):
    model = model_cls(n_channels=1, n_classes=1).to(device)
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        print(f"[✓] Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"[!] Checkpoint not found: {checkpoint_path} — using random weights")
    model.eval()
    return model


@torch.no_grad()
def predict(model, image_tensor: torch.Tensor, device: torch.device,
            threshold: float = 0.5) -> np.ndarray:
    """Return binary mask (H, W) as numpy array."""
    x     = image_tensor.unsqueeze(0).to(device)   # [1,1,H,W]
    logit = model(x)                                # [1,1,H,W]
    prob  = torch.sigmoid(logit).squeeze().cpu().numpy()
    return (prob > threshold).astype(np.uint8)


def denorm(tensor: torch.Tensor) -> np.ndarray:
    """Reverse normalise (mean=0.5, std=0.5) and return HxW numpy array."""
    arr = tensor.squeeze().numpy()
    arr = arr * 0.5 + 0.5          # back to [0,1]
    return np.clip(arr, 0, 1)


# ─────────────────────────────────────────────
# Figure generation
# ─────────────────────────────────────────────

def make_comparison_grid(samples, unet, halfunet, device, output_path):
    """
    Grid: each row = one sample
    Columns: Input | Ground Truth | U-Net | Half U-Net
    """
    n  = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.2 * n))
    if n == 1:
        axes = [axes]

    col_titles = ["Input Image", "Ground Truth", "U-Net", "Half U-Net"]
    col_colors = ["#374151",    "#374151",      "#2563EB", "#DC2626"]

    for col_idx, (title, color) in enumerate(zip(col_titles, col_colors)):
        axes[0][col_idx].set_title(title, fontsize=12, fontweight="bold",
                                   color=color, pad=8)

    for row_idx, (img_t, mask_t) in enumerate(samples):
        img_np  = denorm(img_t)
        mask_np = mask_t.squeeze().numpy()

        pred_unet     = predict(unet,     img_t, device)
        pred_halfunet = predict(halfunet, img_t, device)

        dsc_u = dice_coefficient(
            unet(img_t.unsqueeze(0).to(device)),
            mask_t.unsqueeze(0).to(device)
        )
        dsc_h = dice_coefficient(
            halfunet(img_t.unsqueeze(0).to(device)),
            mask_t.unsqueeze(0).to(device)
        )

        panels = [img_np, mask_np, pred_unet, pred_halfunet]
        cmaps  = ["gray",  "gray",   "gray",    "gray"]

        for col_idx, (panel, cmap) in enumerate(zip(panels, cmaps)):
            ax = axes[row_idx][col_idx]
            ax.imshow(panel, cmap=cmap, vmin=0, vmax=1)
            ax.axis("off")

            # Annotate DSC on prediction columns
            if col_idx == 2:
                ax.set_xlabel(f"DSC: {dsc_u:.4f}", fontsize=9,
                              color="#2563EB", labelpad=2)
                ax.xaxis.set_label_position("bottom")
                ax.xaxis.set_tick_params(labelbottom=True)
            if col_idx == 3:
                ax.set_xlabel(f"DSC: {dsc_h:.4f}", fontsize=9,
                              color="#DC2626", labelpad=2)
                ax.xaxis.set_label_position("bottom")
                ax.xaxis.set_tick_params(labelbottom=True)

        axes[row_idx][0].set_ylabel(f"Sample {row_idx + 1}",
                                    fontsize=10, rotation=90, labelpad=6)

    fig.suptitle(
        "Segmentation Benchmark: U-Net vs Half U-Net\n"
        "Architectural Complexity Reduction in Medical Image Segmentation",
        fontsize=13, fontweight="bold", y=1.01
    )

    # Legend
    patch_u = mpatches.Patch(color="#2563EB", label="U-Net (baseline)")
    patch_h = mpatches.Patch(color="#DC2626", label="Half U-Net (reduced capacity)")
    fig.legend(handles=[patch_u, patch_h], loc="lower center",
               ncol=2, fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[✓] Grid saved → {output_path}")
    plt.close()


def make_overlay(image_np, mask_np, pred_np, model_name, color, output_path):
    """Overlay predicted mask contour on the input image."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    axes[0].imshow(image_np, cmap="gray"); axes[0].set_title("Input",        fontsize=11)
    axes[1].imshow(mask_np,  cmap="gray"); axes[1].set_title("Ground Truth", fontsize=11)

    axes[2].imshow(image_np, cmap="gray")
    overlay = np.zeros((*pred_np.shape, 4))
    overlay[pred_np == 1] = [*plt.cm.colors.to_rgb(color), 0.45]
    axes[2].imshow(overlay)
    axes[2].set_title(f"{model_name} Overlay", fontsize=11, color=color, fontweight="bold")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[✓] Overlay saved → {output_path}")
    plt.close()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Visualise segmentation predictions")
    p.add_argument("--data_dir",   type=str, required=True)
    p.add_argument("--dataset",    type=str, default="busi", choices=["busi","isic"])
    p.add_argument("--img_size",   type=int, default=256)
    p.add_argument("--n_samples",  type=int, default=6,
                   help="Number of samples to visualise")
    p.add_argument("--ckpt_dir",   type=str, default="checkpoints")
    p.add_argument("--output",     type=str, default="results/segmentation_comparison.png")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Load dataset (no augmentation for visualization)
    if args.dataset == "busi":
        dataset = BUSIDataset(args.data_dir, img_size=args.img_size, augment=False)
    else:
        dataset = ISICDataset(args.data_dir, img_size=args.img_size, augment=False)

    indices = random.sample(range(len(dataset)), min(args.n_samples, len(dataset)))
    samples = [dataset[i] for i in indices]

    # Load models
    unet     = load_model(UNet,     f"{args.ckpt_dir}/UNet_best.pth",     device)
    halfunet = load_model(HalfUNet, f"{args.ckpt_dir}/HalfUNet_best.pth", device)

    # Comparison grid
    make_comparison_grid(samples, unet, halfunet, device, args.output)

    # Overlays for first sample
    img_t, mask_t = samples[0]
    img_np  = denorm(img_t)
    mask_np = mask_t.squeeze().numpy()

    with torch.no_grad():
        pred_u = predict(unet,     img_t, device)
        pred_h = predict(halfunet, img_t, device)

    make_overlay(img_np, mask_np, pred_u, "U-Net",     "#2563EB",
                 "results/overlay_unet.png")
    make_overlay(img_np, mask_np, pred_h, "Half U-Net","#DC2626",
                 "results/overlay_halfunet.png")

    print("\n[Done] All visualisations saved to results/")


if __name__ == "__main__":
    main()
