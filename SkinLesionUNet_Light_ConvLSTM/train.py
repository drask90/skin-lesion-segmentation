# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import json
import torch
import numpy as np
from model import UNet
from data_preparation import get_fold_loaders


def bce_iou_loss(outputs, targets, eps=1e-8):
    bce = torch.nn.functional.binary_cross_entropy(outputs, targets)
    intersection = (outputs * targets).sum(dim=(1, 2, 3))
    union = (outputs + targets - outputs * targets).sum(dim=(1, 2, 3))
    iou = (intersection / (union + eps)).mean()
    return bce + (1 - iou)


def train_fold(fold, device):
    model = UNet().to(device)
    train_loader, val_loader = get_fold_loaders(fold)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    best_dsc = 0.0
    best_iou = 0.0
    best_sensitivity = 0.0
    best_specificity = 0.0
    best_accuracy = 0.0
    start_epoch = 0

    checkpoint_path = f'checkpoint_fold{fold+1}.pth'
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_dsc = ckpt['best_dsc']
        best_iou = ckpt['best_iou']
        best_sensitivity = ckpt['best_sensitivity']
        best_specificity = ckpt['best_specificity']
        best_accuracy = ckpt['best_accuracy']
        print(f"Resuming Fold {fold+1} from epoch {start_epoch + 1}")

    for epoch in range(start_epoch, 200):
        model.train()
        for images, masks in train_loader:
            outputs = torch.sigmoid(model(images.to(device)))
            loss = bce_iou_loss(outputs, masks.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0
            iou_total = 0
            dsc_total = 0
            sensitivity_total = 0
            specificity_total = 0
            accuracy_total = 0
            for images, masks in val_loader:
                masks = masks.to(device)
                outputs = torch.sigmoid(model(images.to(device)))
                val_loss += bce_iou_loss(outputs, masks)

                preds = (outputs > 0.5).float()
                intersection = (preds * masks).sum(dim=(1, 2, 3))
                union = (preds + masks).clamp(0, 1).sum(dim=(1, 2, 3))
                iou_total += (intersection / (union + 1e-8)).mean().item()
                dsc_total += (2 * intersection / (preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1e-8)).mean().item()

                tp = (preds * masks).sum(dim=(1, 2, 3))
                tn = ((1 - preds) * (1 - masks)).sum(dim=(1, 2, 3))
                fp = (preds * (1 - masks)).sum(dim=(1, 2, 3))
                fn = ((1 - preds) * masks).sum(dim=(1, 2, 3))

                sensitivity_total += (tp / (tp + fn + 1e-8)).mean().item()
                specificity_total += (tn / (tn + fp + 1e-8)).mean().item()
                accuracy_total += ((tp + tn) / (tp + tn + fp + fn + 1e-8)).mean().item()

        iou = iou_total / len(val_loader)
        dsc = dsc_total / len(val_loader)
        sensitivity = sensitivity_total / len(val_loader)
        specificity = specificity_total / len(val_loader)
        accuracy = accuracy_total / len(val_loader)
        print(f"Fold {fold+1}, Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f}, "
              f"IoU: {iou:.4f}, DSC: {dsc:.4f}, Sen: {sensitivity:.4f}, Spe: {specificity:.4f}, Acc: {accuracy:.4f}")

        if dsc > best_dsc:
            best_dsc = dsc
            best_iou = iou
            best_sensitivity = sensitivity
            best_specificity = specificity
            best_accuracy = accuracy
            torch.save(model.state_dict(), f'best_model_fold{fold+1}.pth')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dsc': best_dsc,
            'best_iou': best_iou,
            'best_sensitivity': best_sensitivity,
            'best_specificity': best_specificity,
            'best_accuracy': best_accuracy,
        }, checkpoint_path)

    os.remove(checkpoint_path)
    return best_dsc, best_iou, best_sensitivity, best_specificity, best_accuracy


def print_model_stats(device):
    model = UNet().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params / 1e6:.2f}M")
    try:
        from thop import profile
        dummy = torch.randn(1, 3, 256, 256).to(device)
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        print(f"FLOPs:      {macs * 2 / 1e9:.2f}G")
    except ImportError:
        print("FLOPs:      install 'thop' to compute  (pip install thop)")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_model_stats(device)

    results_path = 'fold_results.json'
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            saved = json.load(f)
        fold_dscs = saved['fold_dscs']
        fold_ious = saved['fold_ious']
        fold_sensitivities = saved['fold_sensitivities']
        fold_specificities = saved['fold_specificities']
        fold_accuracies = saved['fold_accuracies']
        start_fold = len(fold_dscs)
        print(f"Resuming from fold {start_fold + 1} ({start_fold} fold(s) already completed)")
    else:
        fold_dscs = []
        fold_ious = []
        fold_sensitivities = []
        fold_specificities = []
        fold_accuracies = []
        start_fold = 0

    for fold in range(start_fold, 5):
        print(f"\n{'='*50}")
        print(f"Training Fold {fold+1}/5")
        print(f"{'='*50}")
        best_dsc, best_iou, best_sensitivity, best_specificity, best_accuracy = train_fold(fold, device)
        fold_dscs.append(best_dsc)
        fold_ious.append(best_iou)
        fold_sensitivities.append(best_sensitivity)
        fold_specificities.append(best_specificity)
        fold_accuracies.append(best_accuracy)

        with open(results_path, 'w') as f:
            json.dump({
                'fold_dscs': fold_dscs,
                'fold_ious': fold_ious,
                'fold_sensitivities': fold_sensitivities,
                'fold_specificities': fold_specificities,
                'fold_accuracies': fold_accuracies,
            }, f)

    os.remove(results_path)

    print(f"\n{'='*50}")
    print("5-Fold Cross Validation Results:")
    print(f"{'='*50}")
    for i in range(5):
        print(f"Fold {i+1}: DSC = {fold_dscs[i]:.4f}, IoU = {fold_ious[i]:.4f}, "
              f"Sen = {fold_sensitivities[i]:.4f}, Spe = {fold_specificities[i]:.4f}, Acc = {fold_accuracies[i]:.4f}")
    print(f"\nMean DSC: {np.mean(fold_dscs):.4f} ± {np.std(fold_dscs):.4f}")
    print(f"Mean IoU: {np.mean(fold_ious):.4f} ± {np.std(fold_ious):.4f}")
    print(f"Mean Sensitivity: {np.mean(fold_sensitivities):.4f} ± {np.std(fold_sensitivities):.4f}")
    print(f"Mean Specificity: {np.mean(fold_specificities):.4f} ± {np.std(fold_specificities):.4f}")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")


if __name__ == "__main__":
    main()
