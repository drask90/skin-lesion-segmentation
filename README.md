# Lightweight UNet for Skin Lesion Segmentation

> ⚠️ **Work in Progress** — This repository is part of an ongoing research project. Results and code are subject to change.

---

## Overview

This project explores lightweight UNet-based architectures for skin lesion segmentation on the **ISIC 2016** dataset. Starting from a lightweight UNet baseline, we investigate two feature enhancement strategies applied at skip connections:

- **Focal Modulation (FM)** — a lightweight attention-free mechanism that modulates features using local context aggregation
- **Bidirectional ConvLSTM** — captures spatial-sequential dependencies across encoder feature levels

The goal is to achieve competitive segmentation performance while keeping the model lightweight and efficient.

---

## Repository Structure

```
skin-lesion-segmentation/
├── SkinLesionUNet_Light_FM/
│   ├── model.py                  # UNet + Focal Modulation at skip connections
│   ├── train.py                  # Training with 5-fold CV
│   ├── predict.py                # Inference & evaluation
│   ├── data_preparation.py       # Dataset splits
│   ├── qualitative_comparison.py # Visual results
│   └── save_results.py           # Save predictions
│
├── SkinLesionUNet_Light_ConvLSTM/
│   ├── model.py                  # UNet + BiConvLSTM at skip connections
│   ├── train.py
│   ├── predict.py
│   ├── data_preparation.py
│   ├── qualitative_comparison.py
│   └── save_results.py
```

---

## Preliminary Results (ISIC 2016, 5-Fold CV)

| Model | DSC ↑ | IoU ↑ | Sensitivity ↑ | Specificity ↑ | Params |
|---|---|---|---|---|---|
| Baseline UNet | 0.8775 | 0.7934 | 0.9041 | 0.9539 | 1.94M |
| FM-UNet | 0.8812 | 0.7982 | 0.9031 | 0.9491 | 2.06M |
| BiConvLSTM-UNet | 0.8807 | 0.7978 | 0.8953 | 0.9631 | 3.14M |

> These are preliminary results. Further experiments are ongoing.

---

## Dataset

Download **ISIC 2016** from the official archive:
🔗 https://challenge.isic-archive.com/data/#2016

Organize as:
```
data/isic2016/
├── train/
│   ├── images/    ← .jpg images
│   └── masks/     ← .png segmentation masks
└── test/
    ├── images/
    └── masks/
```

> ⚠️ Dataset is not included in this repo due to ISIC terms of use.

---

## Installation

```bash
git clone https://github.com/drask90/skin-lesion-segmentation.git
cd skin-lesion-segmentation
pip install -r requirements.txt
```

---

## Usage

**Train FM-UNet:**
```bash
cd SkinLesionUNet_Light_FM
python train.py --data_dir ../data/isic2016 --epochs 200
```

**Train BiConvLSTM-UNet:**
```bash
cd SkinLesionUNet_Light_ConvLSTM
python train.py --data_dir ../data/isic2016 --epochs 200
```

**Run inference:**
```bash
python predict.py --weights checkpoints/best.pth --input ../data/isic2016/test/images
```

---

## Status

- [x] Lightweight UNet baseline
- [x] Focal Modulation at skip connections
- [x] BiConvLSTM at skip connections
- [ ] Extended experiments on ISIC 2017 / 2018
- [ ] Paper writing in progress

---

## Citation

This work is currently unpublished. Citation details will be added upon acceptance.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
