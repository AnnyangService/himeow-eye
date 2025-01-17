from ultralytics import YOLO

# 학습된 모델 경로
model_path = "/home/minelab/바탕화면/ANN/zoo0o/himeow-eye/models/yolo/base/runs/weights/best.pt"

# YOLO 모델 로드
model = YOLO(model_path)

# 예측할 이미지 경로
image_path = "/home/minelab/바탕화면/ANN/zoo0o/himeow-eye/data/wp_3917859201504692806498.png"

# 예측 수행
results = model.predict(
    source=image_path,  # 예측할 이미지 경로
    imgsz=224,          # 이미지 크기 (학습 시 사용한 imgsz와 동일해야 함)
    save=True,          # 결과를 저장할지 여부
    save_txt=True,      # 결과를 텍스트로 저장할지 여부
    project="/home/minelab/바탕화면/ANN/zoo0o/himeow-eye/models/yolo/base",  # 결과 저장 경로
    name="predictions", # 저장 디렉토리 이름
    verbose=True        # 세부 정보 출력 여부
)

# 결과 확인
print("Predictions:", results)
