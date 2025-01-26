import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import os
from scipy import ndimage
from typing import List, Dict, Union
import glob

"""모든 이미지 인코더 통과 후 기준에 따라 점수 판단 및 top 20 채널 선택"""
class CustomEncoder:
    def __init__(self, model_name="facebook/sam-vit-base", checkpoint_path=None, gpu_id=3,
                 padding_config=None, scoring_config=None):
        # 기본 설정값 정의
        self.padding_config = {
            'threshold': 0.8,    # 패딩 판단 임계값
            'height_ratio': 8    # 패딩 영역 비율 (height // ratio)
        } if padding_config is None else padding_config

        self.scoring_config = {
            'contrast_weight': 0.5,      # contrast score 가중치
            'edge_weight': 0.5,          # edge score 가중치
            'high_percentile': 90,       # contrast 상위 퍼센타일
            'low_percentile': 10,        # contrast 하위 퍼센타일
            'top_k': 30                  # 선택할 상위 채널 수
        } if scoring_config is None else scoring_config

        # GPU 설정
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        # 모델 로드
        self.model = SamModel.from_pretrained(model_name)
        self.processor = SamProcessor.from_pretrained(model_name)
        
        if checkpoint_path:
            self.load_custom_checkpoint(checkpoint_path)
        
        self.model = self.model.to(self.device)
        self.vision_encoder = self.model.vision_encoder
        self.vision_encoder.eval()

    
    def load_custom_checkpoint(self, checkpoint_path):
        """파인튜닝된 sam vision encoder 불러오기기"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        current_state_dict = self.model.state_dict()
        for name, param in checkpoint['vision_encoder_state_dict'].items():
            if 'vision_encoder' in name:
                current_state_dict[name] = param
        self.model.load_state_dict(current_state_dict)
        print(f"Loaded custom checkpoint from {checkpoint_path}")

    def check_padding_activation(self, channel):
        """1. 패딩 영역 활성화 체크"""
        h = channel.shape[0]
        padding_height = h // self.padding_config['height_ratio']
        
        top_pad = channel[:padding_height].mean()
        bottom_pad = channel[-padding_height:].mean()
        center = channel[padding_height:-padding_height].mean()
        
        return (top_pad > center * self.padding_config['threshold']) or \
               (bottom_pad > center * self.padding_config['threshold'])

    def calculate_channel_scores(self, features):
        """2. 채널 스코어 메기기기"""
        scores = []
        padding_excluded = []
        
        for i in range(features.shape[1]):
            channel = features[0, i].cpu().numpy()
            
            # 패딩 체크
            if self.check_padding_activation(channel):
                scores.append(-float('inf'))
                padding_excluded.append(i)
                continue
            
            # a. Contrast score
            high_thresh = np.percentile(channel, self.scoring_config['high_percentile'])
            low_thresh = np.percentile(channel, self.scoring_config['low_percentile'])
            contrast_score = high_thresh - low_thresh
            
            # b. Edge detection score
            grad_x = np.gradient(channel, axis=1)
            grad_y = np.gradient(channel, axis=0)
            edge_score = np.mean(np.sqrt(grad_x**2 + grad_y**2))
            
            # 최종 점수 계산
            final_score = (
                contrast_score * self.scoring_config['contrast_weight'] +
                edge_score * self.scoring_config['edge_weight']
            )
            
            scores.append(final_score)
        
        # 상위 k개 채널 선택
        top_channels = np.argsort(scores)[-self.scoring_config['top_k']:][::-1]
        return top_channels, scores, padding_excluded

    def extract_feature_vector(self, image_path: str) -> np.ndarray:
        """
        이미지를 인코더에 통과시키고 선택된 채널들의 평균 특징맵 추출
        """
        # 이미지 로드 및 처리
        image = Image.open(image_path)
        inputs = self.processor(
            images=image,
            return_tensors="pt",
            do_resize=True,
            size={"longest_edge": 1024},
            do_normalize=True
        )
        
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()}
        
        # 특징 추출 및 채널 선택
        with torch.no_grad():
            features = self.vision_encoder(inputs["pixel_values"]).last_hidden_state
            top_channels, scores, _ = self.calculate_channel_scores(features)
            
            # 선택된 채널들의 특징맵 가져오기
            selected_features = np.stack([
                features[0, idx].cpu().numpy() 
                for idx in top_channels[:self.scoring_config['top_k']]
            ])
            
            # 채널 방향으로 평균 계산 (shape: [H, W])
            mean_feature_map = np.mean(selected_features, axis=0)
            
            # 특징맵을 1차원 벡터로 변환 (shape: [H * W])
            feature_vector = mean_feature_map.flatten()
            
            return feature_vector

    def visualize_channels(self, image_path, save_dir):
        """선택된 채널 시각화"""
        # 이미지 로드 및 처리
        image = Image.open(image_path)
        inputs = self.processor(
            images=image,
            return_tensors="pt",
            do_resize=True,
            size={"longest_edge": 1024},
            do_normalize=True
        )
        
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # 특징 추출 및 채널 선택
        with torch.no_grad():
            features = self.vision_encoder(inputs["pixel_values"]).last_hidden_state
            top_channels, scores, padding_excluded = self.calculate_channel_scores(features)
            
            os.makedirs(save_dir, exist_ok=True)
            
            # 1. 원본 이미지
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.title('Original Image')
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, 'original.png'))
            plt.close()
            
            # 2. 선택된 top k 채널 시각화
            plt.figure(figsize=(25, 20))
            for idx, channel_idx in enumerate(top_channels[:30]):
                plt.subplot(5, 6, idx + 1)
                channel_features = features[0, channel_idx].cpu().numpy()
                plt.imshow(channel_features, cmap='viridis')
                plt.title(f'Channel {channel_idx}\nScore: {scores[channel_idx]:.4f}')
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'selected_channels.png'))
            plt.close()
            
            # 3. 선택된 채널들의 평균 시각화
            plt.figure(figsize=(15, 5))
            
            # 모든 선택된 채널의 평균
            plt.subplot(1, 2, 1)
            selected_features = np.stack([features[0, idx].cpu().numpy() for idx in top_channels[:self.scoring_config['top_k']]])
            mean_features = np.mean(selected_features, axis=0)
            plt.imshow(mean_features, cmap='viridis')
            plt.title(f'Mean of Top {self.scoring_config["top_k"]} Channels')
            plt.axis('off')
            
            # 상위 5개 채널의 평균
            plt.subplot(1, 2, 2)
            top5_features = np.stack([features[0, idx].cpu().numpy() for idx in top_channels[:5]])
            top5_mean_features = np.mean(top5_features, axis=0)
            plt.imshow(top5_mean_features, cmap='viridis')
            plt.title('Mean of Top 5 Channels')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'mean_channels.png'))
            plt.close()
            
            print(f"Results saved in {save_dir}")
            print(f"\nExcluded {len(padding_excluded)} channels due to strong padding activation:")
            print(f"Channels: {padding_excluded}")
            print(f"\nTop {self.scoring_config['top_k']} selected channels and their scores:")
            for i, ch in enumerate(top_channels[:self.scoring_config['top_k']]):
                print(f"Channel {ch}: {scores[ch]:.4f}")
            

def process_directory(image_dir: str, save_path: str, **kwargs) -> Dict[str, np.ndarray]:
    """
    디렉토리 내의 모든 이미지를 처리하여 특징 벡터를 추출
    """
    visualizer = CustomEncoder(**kwargs)
    
    # 지원하는 이미지 확장자
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    
    results = {}
    
    # 모든 이미지 파일 찾기
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
    
    
    # 각 이미지 처리
    print(f"Found {len(image_files)} images to process")
    for i, image_path in enumerate(image_files, 1):
        try:
            # 진행상황 출력 (같은 줄에 업데이트)
            print(f"\rProcessing image {i}/{len(image_files)}: {os.path.basename(image_path)}", end='', flush=True)
            
            # 특징 벡터 추출
            feature_vector = visualizer.extract_feature_vector(image_path)
            
            # 상대 경로로 저장 (키로 사용)
            rel_path = os.path.relpath(image_path, image_dir)
            results[rel_path] = feature_vector
            
        except Exception as e:
            print(f"\nError processing {image_path}: {str(e)}")

    # 처리 완료 메시지
    print("\nFeature extraction completed!")
    
    # 결과 저장
    np.save(save_path, results)
    
    return results
            
                
if __name__ == "__main__":
    # 설정
    config = {
        'checkpoint_path': "/home/minelab/desktop/ANN/jojun/himeow-eye/models/encoder/finetuning/custom_models/best_checkpoint.pth",  # 체크포인트 경로 설정
        'padding_config': {
            'threshold': 0.7,
            'height_ratio': 10
        },
        'scoring_config': {
            'contrast_weight': 0.5,
            'edge_weight': 0.5,
            'high_percentile': 95,
            'low_percentile': 5,
            'top_k': 20
        }
    }
    
    # 입력 디렉토리와 결과 저장 경로 설정
    input_dir = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/keratitis"
    output_path = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/vectors/origin.npy"
    
    # 디렉토리 처리
    feature_vectors = process_directory(
        image_dir=input_dir,
        save_path=output_path,
        **config
    )
    