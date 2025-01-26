import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

class CustomSamEncoder:
    def __init__(self, model_name="facebook/sam-vit-base", checkpoint_path=None, gpu_id=3):
        # GPU 설정
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        # SAM 모델과 프로세서 로드
        self.model = SamModel.from_pretrained(model_name)
        self.processor = SamProcessor.from_pretrained(model_name)

        # 체크포인트 로드
        if checkpoint_path:
            self.load_custom_checkpoint(checkpoint_path)

        # 모델을 GPU로 이동
        self.model = self.model.to(self.device)
        
        # 비전 인코더 추출 및 평가 모드 설정
        self.vision_encoder = self.model.vision_encoder
        self.vision_encoder.eval()

    def load_custom_checkpoint(self, checkpoint_path):
        """
        파인튜닝된 모델의 체크포인트를 로드합니다.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 체크포인트의 vision_encoder 가중치만 로드
        current_state_dict = self.model.state_dict()
        for name, param in checkpoint['vision_encoder_state_dict'].items():
            if 'vision_encoder' in name:
                current_state_dict[name] = param
        
        self.model.load_state_dict(current_state_dict)
        print(f"Loaded custom checkpoint from {checkpoint_path}")

    def process_image(self, image_path, save_dir=None):
        """
        이미지를 입력으로 받아서 인코더를 통과시키고 결과를 시각화합니다.
        Args:
            image_path (str): 이미지 파일 경로
            save_dir (str): 결과 저장 디렉토리 (기본값: None)
        Returns:
            torch.Tensor: 인코더 출력값
        """
        # 이미지 로드 및 전처리
        image = Image.open(image_path)
        inputs = self.processor(
            images=image,
            return_tensors="pt",
            do_resize=True,
            size={"longest_edge": 1024},
            do_normalize=True
        )
        
        # GPU로 이동
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # 인코더 통과
        with torch.no_grad():
            outputs = self.vision_encoder(inputs["pixel_values"])
            
            # 시각화를 위해 CPU로 이동
            features = outputs.last_hidden_state.cpu()
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                
                # 1. 원본 이미지와 평균 feature map
                plt.figure(figsize=(15, 5))
                
                plt.subplot(121)
                plt.imshow(image)
                plt.title('Original Image')
                plt.axis('off')
                
                plt.subplot(122)
                feature_mean = features[0].mean(dim=0)
                plt.imshow(feature_mean.numpy(), cmap='viridis')
                plt.title('Average Feature Map')
                plt.colorbar()
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'feature_map_avg.png'))
                plt.close()
                
                # 2. 개별 채널 시각화 (처음 64개)
                num_channels = 64
                rows = 8
                cols = 8
                plt.figure(figsize=(20, 20))
                
                for i in range(num_channels):
                    plt.subplot(rows, cols, i + 1)
                    plt.imshow(features[0, i].numpy(), cmap='viridis')
                    plt.title(f'Channel {i}')
                    plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'feature_map_channels.png'))
                plt.close()
            
        return outputs

if __name__ == "__main__":
    # 설정
    checkpoint_path = "/home/minelab/desktop/ANN/jojun/himeow-eye/models/encoder/finetuning/custom_models/best_checkpoint.pth"  # 파인튜닝된 모델의 체크포인트 경로
    save_dir = "/home/minelab/desktop/ANN/jojun/himeow-eye/assets/encoder_fintuned_test_result"  # 결과 저장 경로
    test_image = "/home/minelab/desktop/ANN/jojun/himeow-eye/assets/encoder_test_dataset/origin2.jpg"  # 테스트할 이미지 경로
    
    # 커스텀 인코더 초기화 및 테스트
    encoder = CustomSamEncoder(
        model_name="facebook/sam-vit-base",  # 기본 모델
        checkpoint_path=checkpoint_path,  # 파인튜닝된 체크포인트
        gpu_id=3  # GPU 설정
    )
    
    # 이미지 처리 및 특징 추출
    outputs = encoder.process_image(test_image, save_dir=save_dir)
    print("Encoder output shape:", outputs.last_hidden_state.shape)