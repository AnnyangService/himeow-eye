import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append('/home/minelab/desktop/ANN/jojun/himeow-eye')
from models.encoder.filtering.extract import FeatureExtractor
from models.encoder.filtering.channel_selection.select import ChannelSelector

def visualize_features(image_path: str, save_dir: str, **kwargs):
    """
    지정된 이미지의 feature map 시각화
    
    Args:
        image_path: 처리할 이미지 파일 경로
        save_dir: 시각화 결과를 저장할 디렉토리 경로
        **kwargs: 설정값들 (checkpoint_path, gpu_id, padding_config, scoring_config)
    """
    # 특징 추출기와 채널 선택기 초기화
    extractor = FeatureExtractor(
        checkpoint_path=kwargs.get('checkpoint_path'),
        gpu_id=kwargs.get('gpu_id', 3)
    )
    
    selector = ChannelSelector(
        padding_config=kwargs.get('padding_config'),
        scoring_config=kwargs.get('scoring_config')
    )
    
    try:
        print(f"Processing image: {os.path.basename(image_path)}")
        
        # 1. 특징 추출
        features = extractor.extract_features(image_path)
        
        # 2. 채널 선택 및 점수 계산
        top_channels, scores, _ = selector.calculate_channel_scores(features)
        
        # 결과 저장 디렉토리 생성
        img_save_dir = os.path.join(save_dir, os.path.splitext(os.path.basename(image_path))[0])
        os.makedirs(img_save_dir, exist_ok=True)
        
        # 시각화 시작
        # 1. 원본 이미지
        image = Image.open(image_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        plt.savefig(os.path.join(img_save_dir, 'original.png'))
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
        plt.savefig(os.path.join(img_save_dir, 'selected_channels.png'))
        plt.close()
        
        # 3. 선택된 채널들의 평균 시각화
        plt.figure(figsize=(15, 5))
        
        # 모든 선택된 채널의 평균
        plt.subplot(1, 2, 1)
        selected_features = np.stack([
            features[0, idx].cpu().numpy() 
            for idx in top_channels[:selector.scoring_config['top_k']]
        ])
        mean_features = np.mean(selected_features, axis=0)
        plt.imshow(mean_features, cmap='viridis')
        plt.title(f'Mean of Top {selector.scoring_config["top_k"]} Channels')
        plt.axis('off')
        
        # 상위 5개 채널의 평균
        plt.subplot(1, 2, 2)
        top5_features = np.stack([
            features[0, idx].cpu().numpy() 
            for idx in top_channels[:5]
        ])
        top5_mean_features = np.mean(top5_features, axis=0)
        plt.imshow(top5_mean_features, cmap='viridis')
        plt.title('Mean of Top 5 Channels')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(img_save_dir, 'mean_channels.png'))
        plt.close()
        
        print("Visualization completed!")
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

if __name__ == "__main__":
    # 설정
    config = {
        'checkpoint_path': "/home/minelab/desktop/ANN/jojun/himeow-eye/models/encoder/finetuning/custom_models/best_checkpoint.pth",
        'gpu_id': 3,
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
    
    # 파일 경로 설정
    image_path = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/other_diseases/crop_C0_0ec37892-60a5-11ec-8402-0a7404972c70.jpg"  # 처리할 이미지 파일 경로
    save_dir = "/home/minelab/desktop/ANN/jojun/himeow-eye/test/encoder_test/featuremaps/final"
    
    # 시각화 실행
    visualize_features(
        image_path=image_path,
        save_dir=save_dir,
        **config
    )