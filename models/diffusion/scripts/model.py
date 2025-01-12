from diffusers import UNet2DModel

def get_model(image_size=128):
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512),  # 길이를 5로 수정
        down_block_types=(
            "DownBlock2D",  # 다운샘플링 블록
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # 어텐션이 포함된 다운샘플링 블록
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # 업샘플링 블록
            "AttnUpBlock2D",  # 어텐션이 포함된 업샘플링 블록
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model