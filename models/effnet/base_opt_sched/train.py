import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, accuracy_score

# 데이터 경로 설정
data_dir = "/home/minelab/desktop/ANN/zoo0o/himeow-eye/data/datasets_yolo/split_by_brachy_datasets/datasets"
train_dir = f"{data_dir}/train"
val_dir = f"{data_dir}/val"
test_dir = f"{data_dir}/test"

# 하이퍼파라미터 설정
model_type = "efficientnet-b0"  # EfficientNet 모델 타입
epochs = 100
batch_size = 16
img_size = 640  # EfficientNet에서 요구하는 입력 크기
learning_rate = 0.001
num_classes = 2
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
output_dir = "/home/minelab/desktop/ANN/zoo0o/himeow-eye/models/effnet/base_opt_sched/runs"
os.makedirs(output_dir, exist_ok=True)

# 데이터셋 전처리
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 데이터 로드
train_dataset = datasets.ImageFolder(train_dir, data_transforms['train'])
val_dataset = datasets.ImageFolder(val_dir, data_transforms['val'])
test_dataset = datasets.ImageFolder(test_dir, data_transforms['test'])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# EfficientNet 모델 정의
model = EfficientNet.from_pretrained(model_type, num_classes=num_classes)
model = model.to(device)

# 손실 함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam 옵티마이저 사용

# 학습 함수
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs, scheduler=None):
    best_acc = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 10)

        # 각 epoch마다 학습 및 검증 단계
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
                dataloader = train_loader
            else:
                model.eval()  # 모델을 평가 모드로 설정
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # 데이터 반복
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 순전파
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 역전파 + 옵티마이저 단계 (학습 단계에서만)
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # 통계 업데이트
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # 모델 복사 (검증 정확도가 더 높을 때만)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_efficientnet_model.pth'))

        # 학습 단계가 끝난 후 학습률 스케줄러 업데이트
        if scheduler and phase == 'train':
            scheduler.step()

    print(f"Best val Acc: {best_acc:.4f}")

# 평가 및 결과 저장 함수
def evaluate_and_save_results(model, dataloader, dataset_name, output_dir):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    results = {
        "accuracy": acc,
        "confusion_matrix": conf_matrix.tolist()
    }

    # JSON 저장
    with open(f"{output_dir}/{dataset_name}_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # 혼동 행렬 시각화 및 저장
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["normal", "abnormal"], yticklabels=["normal", "abnormal"])
    plt.title(f"{dataset_name.capitalize()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset_name}_confusion_matrix.png")
    plt.close()

# 학습률 스케줄러 추가
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 10 epoch마다 학습률 감소

# 모델 학습
train_model(model, criterion, optimizer, train_loader, val_loader, epochs, scheduler)

# 테스트 실행 및 결과 저장
model.load_state_dict(torch.load(os.path.join(output_dir, 'best_efficientnet_model.pth'), weights_only=True))
evaluate_and_save_results(model, val_loader, "validation", output_dir)
evaluate_and_save_results(model, test_loader, "test", output_dir)
