import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import os

class ChannelVisualizer:
    def __init__(self, model_name="facebook/sam-vit-base", checkpoint_path=None, gpu_id=3):
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.model = SamModel.from_pretrained(model_name)
        self.processor = SamProcessor.from_pretrained(model_name)
        
        if checkpoint_path:
            self.load_custom_checkpoint(checkpoint_path)
        
        self.model = self.model.to(self.device)
        self.vision_encoder = self.model.vision_encoder
        self.vision_encoder.eval()

    def load_custom_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        current_state_dict = self.model.state_dict()
        for name, param in checkpoint['vision_encoder_state_dict'].items():
            if 'vision_encoder' in name:
                current_state_dict[name] = param
        self.model.load_state_dict(current_state_dict)
        print(f"Loaded custom checkpoint from {checkpoint_path}")

    def check_padding_activation(self, channel, padding_threshold=0.8):
        """위아래 패딩 영역의 활성화를 체크"""
        h = channel.shape[0]
        padding_height = h // 8  # 위아래 약 12.5% 영역을 패딩으로 간주
        
        top_pad = channel[:padding_height].mean()
        bottom_pad = channel[-padding_height:].mean()
        center = channel[padding_height:-padding_height].mean()
        
        # 패딩 영역의 평균 활성화가 중앙 영역보다 큰 경우 True 반환
        return (top_pad > center * padding_threshold) or (bottom_pad > center * padding_threshold)

    def calculate_channel_scores(self, features, top_k=30):
        """각 채널의 contrast와 clarity를 평가하고 패딩 활성화 체크"""
        scores = []
        padding_excluded = []
        
        for i in range(features.shape[1]):
            channel = features[0, i].cpu().numpy()
            
            # 패딩 활성화 체크
            if self.check_padding_activation(channel):
                scores.append(-float('inf'))  # 패딩이 강하게 활성화된 채널은 제외
                padding_excluded.append(i)
                continue
            
            # Contrast score: 상위 20%와 하위 20% 값의 차이
            high_thresh = np.percentile(channel, 80)
            low_thresh = np.percentile(channel, 20)
            contrast_score = high_thresh - low_thresh
            
            # Clarity score: 값의 표준편차
            clarity_score = np.std(channel)
            
            # 최종 점수
            final_score = contrast_score * clarity_score
            scores.append(final_score)
        
        # 상위 k개 채널 선택 (패딩 활성화 채널 제외)
        top_channels = np.argsort(scores)[-top_k:][::-1]
        return top_channels, scores, padding_excluded

    def visualize_channels(self, image_path, save_dir):
        """이미지의 채널별 특징을 추출하고 시각화"""
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
            
            # 2. 선택된 상위 채널 시각화 (5x6 그리드)
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
            
            # 3. 모든 채널 시각화 (8x8 그리드)
            plt.figure(figsize=(30, 30))
            for i in range(64):
                plt.subplot(8, 8, i + 1)
                channel_features = features[0, i].cpu().numpy()
                plt.imshow(channel_features, cmap='viridis')
                
                title_color = 'red' if i in top_channels[:30] else 'black'
                if i in padding_excluded:
                    title_color = 'gray'  # 패딩이 강한 채널은 회색으로 표시
                
                plt.title(f'Channel {i}\nScore: {scores[i]:.4f}', color=title_color)
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'all_channels.png'))
            plt.close()
            
            print(f"Results saved in {save_dir}")
            print(f"\nExcluded {len(padding_excluded)} channels due to strong padding activation:")
            print(f"Channels: {padding_excluded}")
            print("\nTop 30 selected channels and their scores:")
            for i, ch in enumerate(top_channels[:30]):
                print(f"Channel {ch}: {scores[ch]:.4f}")

if __name__ == "__main__":
    # 설정
    checkpoint_path = "/home/minelab/desktop/ANN/jojun/himeow-eye/models/encoder/finetuning/custom_models/best_checkpoint.pth"
    save_dir = "/home/minelab/desktop/ANN/jojun/himeow-eye/assets/encoder_fintuned_test_result2"
    image_path = "/home/minelab/desktop/ANN/jojun/himeow-eye/assets/encoder_test_dataset/origin2.jpg"
    
    # 실행
    visualizer = ChannelVisualizer(checkpoint_path=checkpoint_path)
    visualizer.visualize_channels(image_path, save_dir)