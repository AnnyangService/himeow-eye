import cv2
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

def extract_blue_mask(img):
    """파란색 마스크를 추출하는 함수"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    kernel = np.ones((3,3), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    
    binary_mask = (blue_mask > 0).astype(np.uint8)
    
    return binary_mask

def convert_masks_to_tiff(input_dir, output_dir):
    """디렉토리 내의 모든 jpg 마스크를 tiff로 변환하고 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    mask_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    
    for mask_file in tqdm(mask_files, desc="Converting masks"):
        # 파일 경로
        input_path = os.path.join(input_dir, mask_file)
        output_path = os.path.join(output_dir, mask_file.replace('.jpg', '.tiff'))
        
        # 이미지 로드 및 마스크 추출
        img = cv2.imread(input_path)
        binary_mask = extract_blue_mask(img)
        
        # PIL Image로 변환 및 리사이즈
        pil_mask = Image.fromarray(binary_mask)
        pil_mask = pil_mask.resize((256, 256))
        
        # TIFF로 저장
        pil_mask.save(output_path)

def visualize_binary_mask(mask_path, save_path=None):
   """Binary mask만 시각화하고 저장하는 함수"""
   # Binary mask 로드
   mask = np.array(Image.open(mask_path))
   
   # 이미지 생성
   plt.figure(figsize=(5, 5))
   plt.imshow(mask, cmap='gray')
   plt.axis("off")
   
   # 여백 제거
   plt.gca().set_position([0, 0, 1, 1])
   
   # 결과 저장
   if save_path:
       plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
       
   plt.close()

# 시각화 실행
mask_path = '/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/other_diseases_gtmasks/tiff_masks/crop_C0_3e0adbae-60a5-11ec-8402-0a7404972c70.tiff'
visualization_path = '/home/minelab/desktop/ANN/visualization.png'

visualize_binary_mask(mask_path, visualization_path)