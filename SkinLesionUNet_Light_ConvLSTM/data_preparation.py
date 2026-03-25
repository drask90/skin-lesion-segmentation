# -*- coding: utf-8 -*-


import os
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2


class ISICDataset(Dataset):
    def __init__(self, img_paths, mask_paths, augment=False):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.augment = augment

        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15)
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        image = self.image_transform(image)
        if self.augment:
            image = self.aug_transform(image)
        mask = self.mask_transform(mask)

        return image, mask


def get_fold_loaders(fold, n_splits=5, batch_size=8):
    all_img_paths = []
    all_mask_paths = []

    for split in ['train', 'test']:
        img_dir = f"data/isic2016/{split}/images"
        mask_dir = f"data/isic2016/{split}/masks"
        for fname in sorted(os.listdir(img_dir)):
            stem = os.path.splitext(fname)[0]
            all_img_paths.append(os.path.join(img_dir, fname))
            all_mask_paths.append(os.path.join(mask_dir, f"{stem}_Segmentation.png"))

    all_img_paths = np.array(all_img_paths)
    all_mask_paths = np.array(all_mask_paths)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(kf.split(all_img_paths))
    train_idx, val_idx = splits[fold]

    train_dataset = ISICDataset(all_img_paths[train_idx], all_mask_paths[train_idx], augment=True)
    val_dataset = ISICDataset(all_img_paths[val_idx], all_mask_paths[val_idx], augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader
