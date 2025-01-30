# from datasets import load_dataset # Hugging Face Datasets 라이브러리의 기능
import os
from PIL import Image

import torch
from scripts.config import config
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

# 로컬 폴더에 있는 데이터셋 활용

data_dir = "/home/minelab/desktop/ANN/고양이/안구/일반/비궤양성각막염/유"

# 이미지 전처리
preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
#       transforms.RandomRotation(degrees=30), # 이미지를 -30도에서 30도 사이로 회전
        transforms.RandomVerticalFlip(p=0.3), # 이미지를 상하로 뒤집는 작업, 확률은 0.3

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

train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)