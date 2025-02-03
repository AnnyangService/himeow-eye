from diffusers import UNet2DModel
import torch

def get_model(image_size=128):
    model = UNet2DModel(
        sample_size=image_size,  # 입력 이미지 크기 (128x128, 256x256 등)
        in_channels=3,  # RGB 이미지 입력 (3 채널)
        out_channels=3,  # 출력도 RGB 이미지 (3 채널)
        layers_per_block=3,  # 각 블록당 ResNet 개수 증가
        block_out_channels=(128, 256, 512, 1024),  # 다운샘플링 후 채널 증가
        down_block_types=(
            "DownBlock2D",       # 일반적인 ResNet 다운샘플링 블록
            "AttnDownBlock2D",   # Self-Attention 적용된 다운샘플링 블록
            "DownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",  # Self-Attention 적용된 업샘플링 블록
            "UpBlock2D",
            "UpBlock2D",
        ),
        norm_num_groups=32,  # Group Normalization 적용
        norm_eps=1e-5,  # Normalization 안정화
        dropout=0.1,  # 과적합 방지를 위한 Dropout
        act_fn="silu",  # 활성화 함수 (SiLU = Swish)
        downsample_type="conv",  # 다운샘플링 방식: CNN 기반
        upsample_type="conv",  # 업샘플링 방식: CNN 기반
        center_input_sample=True,  # 입력 이미지를 -1~1 범위로 정규화
    )
    return model

# 모델 생성 및 테스트
unet_model = get_model(image_size=128)

# 입력 텐서 생성 (128x128 RGB 이미지 배치)
dummy_input = torch.randn(1, 3, 128, 128)

# 모델 추론 실행
output = unet_model(dummy_input, timestep=10)

# 출력 이미지 확인
print("Output shape:", output.sample.shape)  # (1, 3, 128, 128)
