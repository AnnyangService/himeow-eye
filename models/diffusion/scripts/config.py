#학습 파라미터들을 포함한 TrainingConfig 클래스

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 128  # 생성되는 이미지 해상도
    train_batch_size = 16
    eval_batch_size = 16  # 평가 동안에 샘플링할 이미지 수
    gradient_accumulation_steps = 1
    #작은 배치크기로도 충분하면 1을 사용할수도 있지만 모델이 큰 배치를 필요로 하고 GPU 메모리가 부족한경우 2, 4 를 사용하기도 함, 가짜로 배치사이즈를 늘려주는 효과를 제공
    
    num_epochs = 100 #에포크 수
    num_train_timesteps = 1000 #타임스텝의 수
    save_image_epochs = 5
    save_model_epochs = 5
    
    learning_rate = 1e-4
    lr_warmup_steps = 500

    mixed_precision = "fp16"  # `no`는 float32, 자동 혼합 정밀도를 위한 `fp16`
    
    output_dir = "/home/minelab/desktop/ANN/Taehwa2/himeow-eye/models/diffusion/output/"  # sample 이미지 저장 경로
    model_save_dir = "/home/minelab/desktop/ANN/Taehwa2/himeow-eye/models/diffusion/output/model" #모델 관련 저장경로
    
    seed = 0 # evaluate.py에서 사용
    
config = TrainingConfig()