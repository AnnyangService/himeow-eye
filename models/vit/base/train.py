from transformers import ViTForImageClassification, Trainer, TrainingArguments
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import torch
import json
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 경로 설정
data_dir = "/home/minelab/desktop/ANN/zoo0o/himeow-eye/data/datasets_yolo/split_by_brachy_datasets/datasets"
output_dir = "/home/minelab/desktop/ANN/zoo0o/himeow-eye/models/vit/base/runs"

# 데이터셋 준비
def preprocess_images(image):
    """PIL 이미지 객체를 전처리합니다."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform(image)

# 데이터셋 로드
train_dataset = load_dataset("imagefolder", data_dir=f"{data_dir}/train")["train"]
val_dataset = load_dataset("imagefolder", data_dir=f"{data_dir}/val")["train"]
test_dataset = load_dataset("imagefolder", data_dir=f"{data_dir}/test")["train"]

# 데이터셋 전처리
train_dataset = train_dataset.map(lambda x: {"pixel_values": preprocess_images(x["image"])}, remove_columns=["image"])
val_dataset = val_dataset.map(lambda x: {"pixel_values": preprocess_images(x["image"])}, remove_columns=["image"])
test_dataset = test_dataset.map(lambda x: {"pixel_values": preprocess_images(x["image"])}, remove_columns=["image"])

# 모델 설정
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", 
    num_labels=2,  # 출력 클래스 수
    ignore_mismatched_sizes=True  # 크기 불일치를 무시
).to("cuda")

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir=f"{output_dir}/results",          # 결과 디렉토리
    eval_strategy="epoch",           # 검증 빈도
    save_strategy="epoch",           # 저장 빈도
    learning_rate=2e-5,              # 학습률
    per_device_train_batch_size=16,  # 배치 크기
    num_train_epochs=100,            # 학습 epoch 수
    weight_decay=0.01,               # 가중치 감소
    logging_dir=f"{output_dir}/logs",            # 로그 디렉토리
    logging_steps=10,                # 로그 기록 간격
    save_total_limit=2,              # 저장 파일 제한
    load_best_model_at_end=True,     # 최고 성능 모델 로드
    metric_for_best_model="accuracy" # 최고 성능 기준
)

# 평가 함수
def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 학습 시작
trainer.train()

# 평가 및 결과 저장 함수
def evaluate_and_save_results(trainer, dataset, dataset_name, output_dir):
    predictions = trainer.predict(dataset)
    pred_logits = predictions.predictions
    pred_labels = np.argmax(pred_logits, axis=1)
    true_labels = predictions.label_ids

    # 정확도 계산
    accuracy = accuracy_score(true_labels, pred_labels)
    # 혼동 행렬 계산
    conf_matrix = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
    # 분류 리포트 생성
    report = classification_report(true_labels, pred_labels, target_names=["normal", "abnormal"], output_dict=True)

    # 결과 저장
    results = {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": report
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

# 평가 및 저장 실행
evaluate_and_save_results(
    trainer=trainer,
    dataset=val_dataset,
    dataset_name="validation",
    output_dir=output_dir
)

evaluate_and_save_results(
    trainer=trainer,
    dataset=test_dataset,
    dataset_name="test",
    output_dir=output_dir
)

# 모델 저장
model.save_pretrained(f"{output_dir}/vit_model")
