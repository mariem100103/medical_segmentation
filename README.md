# Architectural Complexity Reduction in Medical Image Segmentation
### Half U-Net vs Standard U-Net — PyTorch Benchmark

> **Motivation:** Inspired by André et al. (MAGMA, 2025), which demonstrated that reducing
> convolutional channel capacity in U-Net architectures preserves quantitative biomarker
> accuracy while significantly cutting memory overhead. This benchmark reproduces and
> extends that analysis on 2D medical ultrasound images.

---

## Overview

Deep learning models for medical image segmentation are often over-parameterised, leading to high GPU memory consumption, increased latency, and difficult deployment in clinical hardware environments.

This repository evaluates **architectural complexity reduction** by comparing a baseline **Standard U-Net** (~31.4M parameters) against a lightweight **Half U-Net** (~7.8M parameters) on breast ultrasound images (**BUSI**). The goal is to measure the trade-off between segmentation accuracy (Dice Similarity Coefficient) and computational efficiency (parameter count, peak VRAM, and inference speed).

---

## Research Question

> *Can halving the convolutional channel depth of a U-Net encoder–decoder preserve
> clinically meaningful segmentation fidelity while substantially reducing parameter count
> and GPU memory allocation?*

---

## Architectures

| Architecture | Encoder Channels | Parameters | Design Principle |
|---|---|---|---|
| **U-Net** (baseline) | `[64, 128, 256, 512, 1024]` | ~31.4M | Ronneberger et al., 2015 |
| **Half U-Net** (ours) | `[32, 64, 128, 256, 512]` | ~7.8M | Channel reduction (André et al., 2025) |

Both models share identical:
- Encoder–decoder topology with skip connections
- Bilinear upsampling in the decoder
- BatchNorm after every convolution
- Composite loss function: **BCE + Soft Dice**

---

## Loss Function

The composite loss addresses class imbalance common in anatomical ROI segmentation:

```
L_total = L_BCE(logits, y) + L_Dice(logits, y)

         where  L_Dice = 1 - (2·Σ(y·ŷ) + ε) / (Σy + Σŷ + ε)
```

Binary Cross-Entropy handles per-pixel accuracy; Soft Dice penalises region-level mismatch
and is robust to foreground/background imbalance.

---

## Benchmark Results

> Results obtained on the **BUSI dataset** (780 images, benign + malignant categories),
> 80/20 train-validation split, 20 epochs, Adam optimiser (lr=1e-4), image size 256×256.

| Model | Best Val DSC ↑ | Parameters ↓ | Peak VRAM (MB) ↓ | Latency (ms) ↓ |
|---|---|---|---|---|
| U-Net (baseline) | **0.7487** | 31,383,681 | 616.8 MB | 25.32 ms |
| Half U-Net | 0.7266 | **7,849,025** | **181.7 MB** | **7.78 ms** |
| **Δ (Half vs Full)** | −2.21 pp | **−75.0%** | **−70.5%** | **−69.3%** |

**Key finding:** Halving channel capacity reduces parameter count by **75.0%**, peak GPU
VRAM by **70.5%**, and inference latency by **69.3%** (~3.25× faster), with only a 2.21 percentage-point
difference in Dice Similarity Coefficient — confirming that over-parameterised models are
not strictly necessary for precise anatomical segmentation, consistent with André et al. (MAGMA, 2025).

---

## Evaluation Metrics

- **Dice Similarity Coefficient (DSC):** primary segmentation quality metric
  ```
  DSC = (2 · |A ∩ B|) / (|A| + |B|)
  ```
- **Parameter count:** `sum(p.numel() for p in model.parameters() if p.requires_grad)`
- **Peak GPU VRAM:** `torch.cuda.max_memory_allocated()` during a single forward pass
- **Inference latency:** mean over 50 forward passes on a single 256×256 image (ms)

---

## Project Structure

```
medical_segmentation/
├── models.py          # U-Net and Half U-Net architecture definitions
├── losses.py          # BCE + Soft Dice composite loss; hard DSC metric
├── dataset.py         # BUSI / ISIC dataset loaders with joint augmentations
├── train.py           # Training loop, benchmarking, CSV + plot export
├── visualize.py       # Inference visualisation and comparison grid
├── requirements.txt
├── checkpoints/
│   ├── UNet_best.pth
│   └── HalfUNet_best.pth
└── results/
    ├── benchmark_results.csv
    ├── training_curves.png
    ├── segmentation_comparison.png
    ├── overlay_unet.png
    └── overlay_halfunet.png
```

---

## Reproducing the Benchmark

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the BUSI dataset

```
https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
```
Extract into `data/` so the layout matches:
```
data/
├── benign/
├── malignant/
└── normal/
```

### 3. Train both models

```bash
python train.py \
    --data_dir  ./data \
    --dataset   busi \
    --img_size  256 \
    --batch_size 8 \
    --epochs    20 \
    --lr        1e-4
```

Checkpoints saved to `checkpoints/`, metrics to `results/benchmark_results.csv`.

### 4. Visualise predictions

```bash
python visualize.py \
    --data_dir  ./data \
    --dataset   busi \
    --n_samples 6 \
    --output    results/segmentation_comparison.png
```

---

## Augmentation Strategy

Joint spatial augmentations applied identically to image and mask during training:

| Transform | Probability | Parameters |
|---|---|---|
| Horizontal flip | 0.5 | — |
| Vertical flip | 0.5 | — |
| Random rotation | 1.0 | ±15° |
| Brightness / Contrast jitter | 0.5 | factor ∈ [0.8, 1.2] |

---

## References

1. André, R., Martin, S., Trabelsi, A., et al. (2025). *Importance of neural network complexity
   for the automatic segmentation of individual thigh muscles in MRI images from patients
   with neuromuscular diseases.* **MAGMA**.

2. André, R., Martin, S., et al. (2025). *Optimized reconstruction of undersampled Dixon
   sequences using new memory-efficient unrolled deep neural networks: HalfVarNet and
   HalfDIRCN.* **Magnetic Resonance in Medicine**.

3. Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional networks for
   biomedical image segmentation.* **MICCAI**.

---

*Implemented by Mariem Aouani — M.Eng. Data Science & AI, ESPRIT School of Engineering*
