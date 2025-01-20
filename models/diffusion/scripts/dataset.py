# from datasets import load_dataset # Hugging Face Datasets 라이브러리의 기능
import os
from PIL import Image

import torch
from scripts.config import TrainingConfig
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

# 로컬 폴더에 있는 데이터셋 활용

data_dir = "/home/minelab/desktop/ANN/Taehwa/himeow-eye/filtered_by_breeds_datasets/brachy/abnormal"

# 이미지 전처리리
preprocess = transforms.Compose(
    [
        transforms.Resize((TrainingConfig.image_size, TrainingConfig.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

images = []

for img_file in os.listdir(data_dir):
    if img_file.endswith((".jpg", ".png", ".jpeg")):  # 이미지 파일 필터링
        img_path = os.path.join(data_dir, img_file)
        with Image.open(img_path) as img:
            images.append(preprocess(img))

dataset = TensorDataset(torch.stack(images))

train_dataloader = DataLoader(dataset, batch_size=TrainingConfig.train_batch_size, shuffle=True)