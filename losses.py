"""
losses.py — Custom Segmentation Loss Functions
===============================================
Implements the composite loss function used for binary medical image segmentation:

    L_total = L_BCE + (1 − Dice_soft)

Soft Dice loss addresses class imbalance common in anatomical ROI segmentation
(e.g., small lesion area vs. large background), which plain BCE cannot handle.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """
    Soft (differentiable) Dice Loss.

        Dice = (2 * sum(y * ŷ) + ε) / (sum(y) + sum(ŷ) + ε)
        L_Dice = 1 − Dice

    ε = 1e-6 for numerical stability.
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat   = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Composite loss: BCE + Soft Dice.

        L_total = L_BCE(logits, targets) + L_Dice(logits, targets)

    BCE handles per-pixel accuracy; Dice addresses class imbalance
    and penalises region-level mismatch — critical for medical ROI segmentation.
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets) + self.dice(logits, targets)


def dice_coefficient(logits: torch.Tensor,
                     targets: torch.Tensor,
                     threshold: float = 0.5,
                     smooth: float = 1e-6) -> float:
    """
    Hard Dice Similarity Coefficient (DSC) for evaluation.
    Thresholds sigmoid probabilities at `threshold` before computing overlap.
    """
    probs = (torch.sigmoid(logits) > threshold).float()
    probs_flat   = probs.view(-1)
    targets_flat = targets.view(-1)
    intersection = (probs_flat * targets_flat).sum().item()
    dsc = (2.0 * intersection + smooth) / (
        probs_flat.sum().item() + targets_flat.sum().item() + smooth
    )
    return dsc
