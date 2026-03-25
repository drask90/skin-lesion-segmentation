# -*- coding: utf-8 -*-
"""
Qualitative comparison of skin lesion segmentation models on ISIC-2016.

Color coding overlaid on the input image:
  Yellow : True Positive  (TP) – correctly predicted lesion
  Red    : False Negative (FN) – missed region (ground-truth only)
  Green  : False Positive (FP) – predicted healthy skin as lesion

Baseline predictions can be placed in:
    predictions/<ModelName>/ISIC_XXXXXXX.png   (binary 0/255 PNG)

UNet predictions are computed on-the-fly from saved fold weights.

Usage:
    python qualitative_comparison.py
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from model import UNet

# ─────────────────────────────────────────────────────────────
# Global settings
# ─────────────────────────────────────────────────────────────
IMG_SIZE  = (256, 256)   # (W, H) for cv2.resize
ALPHA     = 0.55         # overlay transparency
N_SAMPLES = 5            # number of sample rows in the figure
OUT_PATH  = "qualitative_comparison_isic2016.png"

# RGB colours for overlay
TP_COLOR = np.array([255, 255,   0], dtype=np.float32)   # Yellow
FN_COLOR = np.array([255,   0,   0], dtype=np.float32)   # Red
FP_COLOR = np.array([  0, 210,   0], dtype=np.float32)   # Green


# ─────────────────────────────────────────────────────────────
# Lightweight baseline: U-Net++  (nested skip connections)
# ─────────────────────────────────────────────────────────────
class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class UNetPP(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        nb = base

        def dc(a, b): return _DoubleConv(a, b)

        # Encoder nodes  X_{i,0}
        self.x00 = dc(in_ch, nb)
        self.x10 = dc(nb,    nb*2)
        self.x20 = dc(nb*2,  nb*4)
        self.x30 = dc(nb*4,  nb*8)
        self.x40 = dc(nb*8,  nb*16)

        # Nested nodes
        self.x01 = dc(nb    + nb*2,  nb)
        self.x11 = dc(nb*2  + nb*4,  nb*2)
        self.x21 = dc(nb*4  + nb*8,  nb*4)
        self.x31 = dc(nb*8  + nb*16, nb*8)

        self.x02 = dc(nb    * 2 + nb*2, nb)
        self.x12 = dc(nb*2  * 2 + nb*4, nb*2)
        self.x22 = dc(nb*4  * 2 + nb*8, nb*4)

        self.x03 = dc(nb    * 3 + nb*2, nb)
        self.x13 = dc(nb*2  * 3 + nb*4, nb*2)

        self.x04 = dc(nb    * 4 + nb*2, nb)

        self.up  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Conv2d(nb, 1, 1)

    def forward(self, x):
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))
        x40 = self.x40(self.pool(x30))

        x01 = self.x01(torch.cat([x00, self.up(x10)], 1))
        x11 = self.x11(torch.cat([x10, self.up(x20)], 1))
        x21 = self.x21(torch.cat([x20, self.up(x30)], 1))
        x31 = self.x31(torch.cat([x30, self.up(x40)], 1))

        x02 = self.x02(torch.cat([x00, x01, self.up(x11)], 1))
        x12 = self.x12(torch.cat([x10, x11, self.up(x21)], 1))
        x22 = self.x22(torch.cat([x20, x21, self.up(x31)], 1))

        x03 = self.x03(torch.cat([x00, x01, x02, self.up(x12)], 1))
        x13 = self.x13(torch.cat([x10, x11, x12, self.up(x22)], 1))

        x04 = self.x04(torch.cat([x00, x01, x02, x03, self.up(x13)], 1))
        return self.head(x04)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _load_image_rgb(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, IMG_SIZE)


def _load_mask_binary(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


def _make_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def _infer(model, img_path, device, transform):
    """Run inference; return binary (H,W) uint8 mask."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t   = transform(img).unsqueeze(0).to(device)
    out = model(t)
    pred = (torch.sigmoid(out) > 0.5).float().squeeze().cpu().numpy()
    pred = cv2.resize(pred.astype(np.float32), IMG_SIZE,
                      interpolation=cv2.INTER_NEAREST)
    return (pred > 0.5).astype(np.uint8)


def _load_precomputed(img_stem, folder):
    """
    Try to load a pre-computed binary mask from predictions/<folder>/.
    Accepts files named  <stem>.png | <stem>.jpg | <stem>_Segmentation.png
    Returns uint8 (H,W) binary mask or None.
    """
    base = os.path.join("predictions", folder)
    if not os.path.isdir(base):
        return None
    for suffix in ["", "_Segmentation"]:
        for ext in [".png", ".jpg"]:
            p = os.path.join(base, f"{img_stem}{suffix}{ext}")
            if os.path.exists(p):
                return _load_mask_binary(p)
    return None


def _try_load_model(cls, weight_paths, device):
    """
    Instantiate `cls`, try loading weights from each path in `weight_paths`.
    Returns (model, path) or (None, None).
    """
    for wp in weight_paths:
        if os.path.exists(wp):
            m = cls().to(device)
            try:
                m.load_state_dict(torch.load(wp, map_location=device))
                m.eval()
                print(f"  Loaded {cls.__name__} from '{wp}'")
                return m, wp
            except Exception as e:
                print(f"  Warning: could not load {wp}: {e}")
    return None, None


# ─────────────────────────────────────────────────────────────
# Overlay builder
# ─────────────────────────────────────────────────────────────
def create_color_overlay(image, gt, pred):
    """
    Blend TP / FN / FP colours onto the original image.
      Yellow → TP,  Red → FN,  Green → FP
    """
    overlay = image.copy().astype(np.float32)

    tp = (gt == 1) & (pred == 1)
    fn = (gt == 1) & (pred == 0)
    fp = (gt == 0) & (pred == 1)

    for mask_bool, color in [(tp, TP_COLOR), (fn, FN_COLOR), (fp, FP_COLOR)]:
        overlay[mask_bool] = (overlay[mask_bool] * (1.0 - ALPHA)
                              + color * ALPHA)

    return np.clip(overlay, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────
# Main figure builder
# ─────────────────────────────────────────────────────────────
def build_figure(img_paths, mask_paths, model_configs, n_samples=5,
                 out_path=OUT_PATH):
    """
    model_configs : list of dicts
      Each dict must have:
        'label'  : column header string
        'type'   : 'nn'  → nn.Module  (needs 'model', 'device', 'transform')
                   'pre' → pre-computed (needs 'folder')
    """
    # Pick evenly spaced samples
    step    = max(1, len(img_paths) // n_samples)
    indices = list(range(0, len(img_paths), step))[:n_samples]

    n_cols = 2 + len(model_configs)          # Image | GT | models…
    n_rows = len(indices)

    cell_px = 2.6                            # inches per cell
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_px, n_rows * cell_px),
        squeeze=False,
    )

    # Column titles (row 0 only)
    col_labels = ['Input Image', 'Ground Truth'] + \
                 [cfg['label'] for cfg in model_configs]
    for c, lbl in enumerate(col_labels):
        axes[0][c].set_title(lbl, fontsize=9, fontweight='bold', pad=6)

    for row, idx in enumerate(indices):
        img_path  = img_paths[idx]
        mask_path = mask_paths[idx]
        stem      = os.path.splitext(os.path.basename(img_path))[0]

        image = _load_image_rgb(img_path)
        gt    = _load_mask_binary(mask_path)

        # ── col 0: input image ────────────────────────────────
        ax = axes[row][0]
        ax.imshow(image)
        ax.set_ylabel(stem, fontsize=6.5, rotation=0,
                      labelpad=62, va='center')
        ax.axis('off')

        # ── col 1: ground truth (white on black) ──────────────
        ax = axes[row][1]
        gt_vis = np.zeros((*IMG_SIZE[::-1], 3), dtype=np.uint8)
        gt_vis[gt == 1] = [255, 255, 255]
        ax.imshow(gt_vis)
        ax.axis('off')

        # ── cols 2+: model overlays ───────────────────────────
        for col_off, cfg in enumerate(model_configs):
            ax = axes[row][2 + col_off]

            if cfg['type'] == 'nn':
                pred = _infer(cfg['model'], img_path,
                              cfg['device'], cfg['transform'])
            elif cfg['type'] == 'pre':
                pred = _load_precomputed(stem, cfg['folder'])
                if pred is None:
                    # Show placeholder
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                            fontsize=11, color='gray',
                            transform=ax.transAxes)
                    ax.set_facecolor('#f0f0f0')
                    ax.axis('off')
                    continue
            else:
                ax.axis('off')
                continue

            ax.imshow(create_color_overlay(image, gt, pred))
            ax.axis('off')

    # Legend
    patches = [
        mpatches.Patch(facecolor='yellow', edgecolor='gray',
                       label='True Positive (TP)'),
        mpatches.Patch(facecolor='red',    edgecolor='gray',
                       label='False Negative (FN) – Missed'),
        mpatches.Patch(facecolor='lime',   edgecolor='gray',
                       label='False Positive (FP) – Over-segmented'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, 0.0),
               frameon=True, edgecolor='black')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"\nSaved: '{out_path}'")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
def main():
    # ── 1. Gather test image / mask pairs ─────────────────────
    img_paths, mask_paths = [], []
    for split in ['test', 'train']:
        img_dir  = os.path.join('data', 'isic2016', split, 'images')
        mask_dir = os.path.join('data', 'isic2016', split, 'masks')
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            stem = os.path.splitext(fname)[0]
            mp   = os.path.join(mask_dir, f'{stem}_Segmentation.png')
            if os.path.exists(mp):
                img_paths.append(os.path.join(img_dir, fname))
                mask_paths.append(mp)

    if not img_paths:
        raise FileNotFoundError(
            'No image-mask pairs found under data/isic2016/.')

    print(f'Found {len(img_paths)} image-mask pairs.')

    # ── 2. Device & shared transform ──────────────────────────
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = _make_transform()
    print(f'Device: {device}')

    # ── 3. Build model_configs ─────────────────────────────────
    model_configs = []

    # ── 3c. UNet (ours) ───────────────────────────────────────
    unet_weights = ['best_model.pth'] + \
                   [f'best_model_fold{k}.pth' for k in range(1, 6)]
    unet, unet_path = _try_load_model(UNet, unet_weights, device)
    if unet is not None:
        model_configs.append({
            'label':     'Ours\n(UNet)',
            'type':      'nn',
            'model':     unet,
            'device':    device,
            'transform': transform,
        })
    else:
        print('Warning: No UNet weights found.')
        model_configs.append({
            'label':  'Ours\n(UNet)',
            'type':   'pre',
            'folder': 'UNet',
        })

    # ── 4. Generate figure ────────────────────────────────────
    build_figure(img_paths, mask_paths, model_configs,
                 n_samples=N_SAMPLES, out_path=OUT_PATH)


if __name__ == '__main__':
    main()
