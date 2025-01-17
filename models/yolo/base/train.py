from ultralytics import YOLO

# 데이터 경로 설정
data_dir = "/home/minelab/desktop/ANN/zoo0o/himeow-eye/data/datasets_yolo/split_by_brachy_datasets/datasets"
train_dir = f"{data_dir}/train"
val_dir = f"{data_dir}/val"
test_dir = f"{data_dir}/test"

# 하이퍼파라미터 설정
model_type = "yolov8x-cls"  # YOLOv8 분류 모델 (n: nano, s: small 등 선택 가능)
epochs = 100
batch = 16
imgsz = 640
learning_rate = 0.001
num_classes = 2

# YOLOv8 모델 생성
device = 'cuda:3'  # device 설정
model = YOLO(model_type).to(device)

# 모델 학습
model.train(
    data=data_dir,  # 전체 데이터 디렉토리 경로를 설정 (train, val 포함)
    epochs=epochs,
    batch=batch,  # 'batch_size'
    imgsz=imgsz,  # 이미지 크기
    lr0=learning_rate,
    workers=4,  # 데이터 로드 작업자 수
    project="/home/minelab/desktop/ANN/zoo0o/himeow-eye/models/yolo/base/runs",  # 결과 저장 디렉토리
    name="yolo8x-cls",
)

# Test
test_metrics = model.val(data=data_dir, split='test')
print("Test Results:", test_metrics)

# TorchScript로 저장
model.export(
    format="torchscript",
    simplify=True,
    name="/home/minelab/desktop/ANN/zoo0o/himeow-eye/models/yolo/base/yolo_model.torchscript",
)
