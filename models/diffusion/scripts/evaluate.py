import os
import torch
from PIL import Image


# 샘플 이미지 배열을 그리드 형식으로 합침
# images : PIL.imgae 객체 리스트
# row, cols : 그리드의 행과 열 개수
def make_grid(images, rows, cols):

    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid

# 모델을 사용해 이미지를 생성하고 결과를 저장
def evaluate(config, epoch, pipeline):

    # Sample some images from random noise (backward diffusion process).
    images = pipeline(
        batch_size=config.eval_batch_size, 
        generator=torch.manual_seed(config.seed),
    ).images

    # Create a grid from the sampled images.
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the image grid to the output directory.
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

