# -*- coding: utf-8 -*-


import torch
from torchvision import transforms
from model import UNet
import cv2


def predict(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(image)

    pred = (torch.sigmoid(pred) > 0.5).float().squeeze().cpu().numpy()
    return pred
