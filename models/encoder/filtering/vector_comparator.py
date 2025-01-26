from inference_and_select import CustomEncoder
import numpy as np
import os
from typing import Dict, List, Tuple
import glob

def save_similarities(similarities: List[Tuple[str, float]], f, num_samples: int = 10):
    """상위, 중간, 하위 유사도를 저장하는 헬퍼 함수"""
    total = len(similarities)
    mid_start = total//2 - num_samples//2
    
    # 상위 유사도
    f.write("\nTop Similarities:\n")
    f.write("-" * 50 + "\n")
    for orig_path, sim in similarities[:num_samples]:
        f.write(f"{orig_path}: {sim:.4f}\n")
    
    # 중간 유사도
    f.write("\nMiddle Similarities:\n")
    f.write("-" * 50 + "\n")
    for orig_path, sim in similarities[mid_start:mid_start+num_samples]:
        f.write(f"{orig_path}: {sim:.4f}\n")
    
    # 하위 유사도
    f.write("\nBottom Similarities:\n")
    f.write("-" * 50 + "\n")
    for orig_path, sim in similarities[-num_samples:]:
        f.write(f"{orig_path}: {sim:.4f}\n")

class VectorComparator:
    def __init__(self, origin_vector_path: str, generated_image_dir: str, encoder_config: dict):
        """
        Args:
            origin_vector_path: 원본 이미지들의 벡터가 저장된 .npy 파일 경로
            generated_image_dir: 생성된 이미지들이 있는 디렉토리 경로
            encoder_config: CustomEncoder 설정
        """
        self.origin_vectors = np.load(origin_vector_path, allow_pickle=True).item()
        self.generated_image_dir = generated_image_dir
        self.encoder = CustomEncoder(**encoder_config)
        print(f"Loaded {len(self.origin_vectors)} original vectors")

    def calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """두 벡터 간의 코사인 유사도 계산"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def process_generated_images(self, save_dir: str) -> Dict[str, List[Tuple[str, float]]]:
        """모든 생성 이미지 처리"""
        os.makedirs(save_dir, exist_ok=True)
        results = {}
        
        # 이미지 파일 찾기
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.generated_image_dir, '**', ext), recursive=True))
        
        print(f"Found {len(image_files)} generated images to process")
        
        # 각 이미지 처리
        for i, image_path in enumerate(image_files, 1):
            try:
                print(f"\rProcessing image {i}/{len(image_files)}: {os.path.basename(image_path)}", 
                      end='', flush=True)
                
                # CustomEncoder를 사용하여 특징 벡터 추출
                gen_vector = self.encoder.extract_feature_vector(image_path)
                
                # 모든 원본 이미지와의 유사도 계산
                similarities = []
                for orig_path, orig_vector in self.origin_vectors.items():
                    similarity = self.calculate_similarity(gen_vector, orig_vector)
                    similarities.append((orig_path, similarity))
                
                # 유사도 기준 내림차순 정렬
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # 결과 저장
                rel_path = os.path.relpath(image_path, self.generated_image_dir)
                results[rel_path] = similarities
                
                # 각 이미지별 결과를 텍스트 파일로 저장
                result_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_similarities.txt")
                with open(result_path, 'w') as f:
                    f.write(f"Similarities for {rel_path}:\n")
                    f.write("=" * 50 + "\n")
                    save_similarities(similarities, f)
                
            except Exception as e:
                print(f"\nError processing {image_path}: {str(e)}")
        
        print("\nComparison completed!")
        
        # 전체 결과를 하나의 파일로도 저장
        with open(os.path.join(save_dir, 'all_similarities.txt'), 'w') as f:
            f.write("Generated Image Similarities\n")
            f.write("=" * 50 + "\n\n")
            
            for gen_path, similarities in results.items():
                f.write(f"\nResults for {gen_path}:\n")
                f.write("=" * 50 + "\n")
                save_similarities(similarities, f)
                f.write("\n" + "=" * 50 + "\n")  # 각 이미지 결과 구분선
        
        return results

if __name__ == "__main__":
    # CustomEncoder 설정
    encoder_config = {
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
    origin_vector_path = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/vectors/origin.npy"
    generated_image_dir = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/generated"  # 생성된 이미지 디렉토리
    save_dir = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/vectors/sim"  # 유사도 결과 저장 디렉토리
    
    # 벡터 비교기 초기화 및 실행
    comparator = VectorComparator(origin_vector_path, generated_image_dir, encoder_config)
    results = comparator.process_generated_images(save_dir)