#grid형태의 샘플이미지를 저장

import os
import torch
from PIL import Image


# 샘플 이미지 배열을 그리드 형식으로 합침
# images : PIL.imgae 객체 리스트
# row, cols : 그리드의 행과 열 개수
def make_grid(images, rows, cols):

    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))  #RGB 모드의 빈 이미지 생성성
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h)) #이미지 위치를 계산해서 하나씩 격자에 붙음음
    return grid


# 모델을 사용해 이미지를 생성하고 결과를 저장
def evaluate(config, epoch, pipeline):
    # pipeline : DDPMPipeline 객체를 사용해 랜덤 노이즈에서 이미지를 생성, reverse process 수행
    images = pipeline(
        batch_size=config.eval_batch_size, 
        generator=torch.manual_seed(config.seed), #pyTorch 난수생성기 초기화해서 같은 난수 생성하는 함수
    ).images
    #["sample"] - notebook code대로 입력시 error, github issue 참고하여 수정
    # 이미지 생성결과를 딕셔너리 형태의 출력으로 반환, 이미지 객체들의 리스트
    
    # 왜 달았지 이주석 그때 분명 뭔가 깨우쳤는데
    # "sample": [<PIL.Image>, <PIL.Image>, <PIL.Image>, ...],  # 생성된 이미지
    # "noise": <torch.Tensor>,  # 랜덤 노이즈
    # "timesteps": <torch.Tensor>,  # 타임스텝 정보
    
    image_grid = make_grid(images, rows=4, cols=4)

    test_dir = os.path.join(config.output_dir, "samples_3")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
    #0010.png 형태로 저장 : 4자리수, 빈자리는 0으로 d-정수 형태로