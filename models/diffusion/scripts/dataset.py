import os  # 파일 및 디렉토리 작업을 위한 모듈
from PIL import Image  # 이미지를 열고 처리하기 위한 모듈
from torchvision import transforms  # 데이터 변환(transform)을 위한 PyTorch 모듈
from torch.utils.data import Dataset  # PyTorch의 데이터셋(Dataset) 클래스

# 이미지 데이터를 처리하고 데이터셋 형태로 관리하기 위한 클래스
class ImageDataset(Dataset):
    def __init__(self, image_dir, image_size=128):
        """
        클래스 초기화 메서드
        Args:
            image_dir (str): 이미지 파일이 저장된 디렉토리 경로
            image_size (int): 이미지를 리사이즈할 크기 (정사각형 기준)
        """
        self.image_dir = image_dir  # 이미지 디렉토리 경로 저장
        # 이미지 변환(transform) 파이프라인 정의
        self.transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # 이미지를 지정된 크기로 리사이즈
            transforms.RandomHorizontalFlip(),  # 이미지를 랜덤으로 좌우 반전
            transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
            transforms.Normalize([0.5], [0.5]),  # 텐서 데이터를 정규화 (평균=0.5, 표준편차=0.5)
        ])
        # 이미지 디렉토리에서 파일 목록 가져오기 (png, jpg, jpeg 확장자만)
        self.image_files = [
            os.path.join(image_dir, file)  # 파일 경로 생성
            for file in os.listdir(image_dir)  # 디렉토리 내 모든 파일 탐색
            if file.endswith(('.png', '.jpg', '.jpeg'))  # 이미지 파일 필터링
        ]

    def __len__(self):
        """
        데이터셋의 총 이미지 개수를 반환
        Returns:
            int: 이미지 파일 개수
        """
        return len(self.image_files)  # 이미지 파일 리스트의 길이 반환

    def __getitem__(self, idx):
        """
        주어진 인덱스(idx)에 해당하는 데이터를 반환
        Args:
            idx (int): 데이터 인덱스
        Returns:
            dict: 변환된 이미지 텐서를 포함한 딕셔너리
        """
        image_path = self.image_files[idx]  # 인덱스에 해당하는 이미지 파일 경로 가져오기
        image = Image.open(image_path).convert("RGB")  # 이미지를 열고 RGB 형식으로 변환
        image = self.transforms(image)  # 정의된 변환(transform) 적용
        return {"image": image}  # 변환된 이미지 텐서를 딕셔너리 형태로 반환
