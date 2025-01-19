import logging
from transformers import Trainer, TrainingArguments, ViTForImageClassification, ViTImageProcessor, ViTConfig
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Seaborn 추가
import json
import warnings

# 불필요한 경고 무시
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars")

# 로깅 설정
logging.basicConfig(
    filename="debug.log",  # 디버깅 로그 파일
    level=logging.INFO,     # 로그 레벨 설정
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 데이터셋 경로 설정
data_dir = "/home/minelab/desktop/ANN/zoo0o/himeow-eye/data/datasets_yolo/split_by_brachy_datasets/datasets"

# ViT 모델 및 프로세서 설정
model_name = "google/vit-base-patch16-224-in21k"

# 모델 구성 수정
config = ViTConfig.from_pretrained(
    model_name,
    image_size=640,  # 입력 크기 수정
    num_labels=2,
    id2label={0: "normal", 1: "abnormal"},
    label2id={"normal": 0, "abnormal": 1}
)

# 수정된 설정으로 모델 생성
model = ViTForImageClassification.from_pretrained(
    model_name,
    config=config,
    ignore_mismatched_sizes=True  # 크기 불일치 무시
).to("cuda:3")

# 프로세서 설정
processor = ViTImageProcessor.from_pretrained(
    model_name,
    size={"height": 640, "width": 640}  # 640x640으로 설정
)

# 데이터셋 로드 및 전처리
dataset = load_dataset("imagefolder", data_dir=data_dir)

# 데이터 전처리 함수
def preprocess_function(examples):
    inputs = processor(images=examples["image"], return_tensors="pt")

    if not hasattr(preprocess_function, "logged") or not preprocess_function.logged:
        logging.info(f"Processor outputs shape: {inputs['pixel_values'].shape}")
        preprocess_function.logged = True

    return {
        "pixel_values": inputs["pixel_values"].squeeze(0) if inputs["pixel_values"].dim() == 4 else inputs["pixel_values"],
        "labels": examples["label"]
    }

preprocess_function.logged = False
encoded_dataset = dataset.map(preprocess_function, batched=True)

# 데이터 Collate 함수
def collate_fn(batch):
    try:
        pixel_values = torch.stack([
            torch.tensor(example["pixel_values"]) if isinstance(example["pixel_values"], list) else example["pixel_values"]
            for example in batch
        ])
        labels = torch.tensor([example["labels"] for example in batch])

        if not hasattr(collate_fn, "logged") or not collate_fn.logged:
            logging.info(f"Batch pixel values shape: {pixel_values.shape}")
            logging.info(f"Batch labels shape: {labels.shape}")
            collate_fn.logged = True

    except Exception as e:
        logging.error(f"Error in collate_fn: {e}")
        logging.error(f"Batch data: {batch}")
        raise e

    return {"pixel_values": pixel_values, "labels": labels}

collate_fn.logged = False

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="/home/minelab/desktop/ANN/zoo0o/himeow-eye/models/vit/base/runs",
    eval_strategy="steps",
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=0.001,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    report_to="tensorboard",
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    data_collator=collate_fn,
)

# 모델 학습
trainer.train()

# 평가 및 결과 저장 함수
def evaluate_and_save_results(dataset, dataset_name, output_dir):
    predictions = trainer.predict(dataset)
    pred_logits = predictions.predictions
    pred_labels = np.argmax(pred_logits, axis=1)
    true_labels = predictions.label_ids

    accuracy = accuracy_score(true_labels, pred_labels)
    conf_matrix = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
    report = classification_report(true_labels, pred_labels, target_names=["normal", "abnormal"], output_dict=True)

    results = {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": report
    }

    with open(f"{output_dir}/{dataset_name}_results.json", "w") as f:
        json.dump(results, f, indent=4)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["normal", "abnormal"], yticklabels=["normal", "abnormal"])
    plt.title(f"{dataset_name.capitalize()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset_name}_confusion_matrix.png")
    plt.close()

# 평가 및 저장
evaluate_and_save_results(
    dataset=encoded_dataset["validation"],
    dataset_name="validation",
    output_dir="/home/minelab/desktop/ANN/zoo0o/himeow-eye/models/vit/base/runs"
)

evaluate_and_save_results(
    dataset=encoded_dataset["test"],
    dataset_name="test",
    output_dir="/home/minelab/desktop/ANN/zoo0o/himeow-eye/models/vit/base/runs"
)

# 모델 저장
trainer.save_model("/home/minelab/desktop/ANN/zoo0o/himeow-eye/models/vit/base/runs/vit_model")
