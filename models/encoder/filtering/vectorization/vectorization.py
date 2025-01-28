import os
import glob
import numpy as np
from typing import Dict
import sys

# 원본 코드 경로 추가
sys.path.append('/home/minelab/desktop/ANN/jojun/himeow-eye')
from models.encoder.filtering.channel_selection.select import CustomEncoder

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
    
    # 입력 디렉토리와 결과 저장 경로 설정
    input_dir = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/keratitis"
    output_path = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/vectors/origin.npy"
    
    # 디렉토리 처리
    feature_vectors = process_directory(
        image_dir=input_dir,
        save_path=output_path,
        **config
    )