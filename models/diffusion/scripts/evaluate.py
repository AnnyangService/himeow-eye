from diffusers import DDPMPipeline  # Diffusers 라이브러리에서 DDPMPipeline 가져오기 (DDPM 모델을 사용하기 위한 파이프라인)
from diffusers.utils import make_image_grid  # 생성된 이미지를 그리드로 시각화하기 위한 유틸리티
import os  # 파일 및 디렉토리 작업을 위한 모듈

# 모델 평가 및 샘플 이미지 저장 함수
def evaluate(config, model, noise_scheduler, epoch):
    """
    DDPM 모델을 사용하여 이미지 생성 및 저장.
    Args:
        config: 모델 설정(config) 객체로 평가에 필요한 파라미터를 포함.
        model: UNet 기반 DDPM 모델.
        noise_scheduler: 노이즈 스케줄러로, DDPM의 샘플링 과정에서 사용.
        epoch (int): 현재 학습 에포크 (이미지 저장 파일 이름에 사용).

    """
    # Diffusers 라이브러리의 DDPMPipeline 생성
    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    
    # 샘플 이미지 생성
    images = pipeline(batch_size=config.eval_batch_size).images
    
    # 생성된 이미지를 4x4 형태의 그리드로 정리
    grid = make_image_grid(images, rows=4, cols=4)

    # 샘플 이미지를 저장할 디렉토리 경로 설정
    output_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(output_dir, exist_ok=True)  # 디렉토리가 없으면 생성

    # 그리드 이미지를 지정된 경로에 저장 (파일명은 에포크 번호로 설정)
    grid.save(f"{output_dir}/{epoch:04d}.png")
