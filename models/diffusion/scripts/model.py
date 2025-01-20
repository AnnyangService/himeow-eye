#UNet2DModel 생성하기

from diffusers import UNet2DModel
from scripts.config import TrainingConfig

def get_model(image_size=TrainingConfig.image_size):
    model = UNet2DModel(
        sample_size=image_size, #이미지 해상도 128
        in_channels=3, # 입력 채널 수, RGB 이미지에서 3
        out_channels=3, # 출력 채널 수
        layers_per_block=2,# UNet 블럭당 몇 개의 ResNet 레이어가 사용되는지
        block_out_channels=(128, 128, 256, 256, 512, 512),  # 길이를 5로 수정
        down_block_types=(
            "DownBlock2D",  # 일반적인 ResNet 다운샘플링 블럭
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # spatial self-attention이 포함된 일반적인 ResNet 다운샘플링 블럭
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # 일반적인 ResNet 업샘플링 블럭
            "AttnUpBlock2D",  # spatial self-attention이 포함된 일반적인 ResNet 업샘플링 블럭
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model