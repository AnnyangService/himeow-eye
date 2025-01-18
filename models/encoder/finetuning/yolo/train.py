from ultralytics import YOLO

# pre-trained segmentation 모델 로드
model = YOLO('yolo11x-seg.pt')  # 다른 옵션: yolov8s-seg.pt, yolov8m-seg.pt, yolov8l-seg.pt, yolov8x-seg.pt

# 이미지로 테스트
results = model(source='/home/minelab/desktop/ANN/jojun/himeow-eye/assets/encoder_test_dataset',
                device=3,
                save=True)  # 결과가 runs/segment/predict 폴더에 저장됨