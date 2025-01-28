import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import random
import sys
sys.path.append('/home/minelab/desktop/ANN/jojun/himeow-eye')
from models.encoder.filtering.channel_selection.select import CustomEncoder

def visualize_random_features(input_dir: str, save_dir: str, num_samples: int = 100, **config):
    """
    디렉토리에서 랜덤하게 이미지를 선택하여 feature map 시각화
    
    Args:
        input_dir: 이미지가 있는 디렉토리 경로
        save_dir: 시각화 결과를 저장할 디렉토리 경로
        num_samples: 시각화할 이미지 수 (기본값: 100)
        **config: CustomEncoder에 전달할 설정값들
    """
    # CustomEncoder 초기화
    encoder = CustomEncoder(**config)
    
    # 이미지 파일 찾기
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
    
    # 랜덤하게 이미지 선택
    if len(image_files) > num_samples:
        selected_files = random.sample(image_files, num_samples)
    else:
        selected_files = image_files
        print(f"Warning: Only {len(image_files)} images found in directory")

    # 각 이미지에 대해 feature map 시각화
    for i, image_path in enumerate(selected_files, 1):
        print(f"\rProcessing image {i}/{len(selected_files)}: {os.path.basename(image_path)}", 
              end='', flush=True)
        
        try:
            # 이미지 로드 및 처리
            image = Image.open(image_path)
            inputs = encoder.processor(
                images=image,
                return_tensors="pt",
                do_resize=True,
                size={"longest_edge": 1024},
                do_normalize=True
            )
            
            inputs = {k: v.to(encoder.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # 특징 추출 및 채널 선택
            with torch.no_grad():
                features = encoder.vision_encoder(inputs["pixel_values"]).last_hidden_state
                top_channels, scores, _ = encoder.calculate_channel_scores(features)
                
                # 결과 저장 디렉토리 생성
                img_save_dir = os.path.join(save_dir, f'image_{i}')
                os.makedirs(img_save_dir, exist_ok=True)
                
                # 1. 원본 이미지
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
                    for idx in top_channels[:encoder.scoring_config['top_k']]
                ])
                mean_features = np.mean(selected_features, axis=0)
                plt.imshow(mean_features, cmap='viridis')
                plt.title(f'Mean of Top {encoder.scoring_config["top_k"]} Channels')
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
                
        except Exception as e:
            print(f"\nError processing {image_path}: {str(e)}")
    
    print("\nVisualization completed!")

if __name__ == "__main__":
    # 설정
    config = {
        'checkpoint_path': "/home/minelab/desktop/ANN/jojun/himeow-eye/models/encoder/finetuning/custom_models/best_checkpoint.pth",
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
    
    # 경로 설정
    input_dir = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/generated"
    save_dir = "/home/minelab/desktop/ANN/jojun/himeow-eye/assets/encoder_test/featuremaps/generated"
    
    # 시각화 실행
    visualize_random_features(
        input_dir=input_dir,
        save_dir=save_dir,
        num_samples=100,
        **config
    )