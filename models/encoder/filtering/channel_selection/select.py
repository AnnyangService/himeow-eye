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
