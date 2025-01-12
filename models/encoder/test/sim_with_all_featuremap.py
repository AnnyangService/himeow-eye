import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict
import numpy as np
from encoder_test import TestSamEncoder

class SamSimilarityCalculator:
    """SAM 인코더를 통과시킨 feature map 전체를 사용하여 이미지 간 유사도를 계산"""
    
    def compute_cosine_similarity_full(self, feature1: torch.Tensor, feature2: torch.Tensor) -> torch.Tensor:
        """전체 feature map을 사용하여 두 이미지 간의 코사인 유사도를 계산
        
        Args:
            feature1 (torch.Tensor): 첫 번째 이미지의 feature map [1, 64, 64, 256]
            feature2 (torch.Tensor): 두 번째 이미지의 feature map [1, 64, 64, 256]
            
        Returns:
            torch.Tensor: 위치별 코사인 유사도의 평균값 (scalar)
        """
        # feature map 형태 변환: [1, 64, 64, 256] -> [4096, 256]
        f1 = feature1.reshape(-1, feature1.size(-1))
        f2 = feature2.reshape(-1, feature2.size(-1))
        
        # L2 정규화 적용
        f1_norm = F.normalize(f1, p=2, dim=1)
        f2_norm = F.normalize(f2, p=2, dim=1)
        
        # 위치별 코사인 유사도 계산
        similarity = torch.mm(f1_norm, f2_norm.t())
        
        # 대각 요소의 평균 계산 (위치별 유사도의 평균)
        return similarity.diag().mean()
    
    def compute_batch_similarities(self, 
                              generated_features: List[torch.Tensor], 
                              original_features: List[torch.Tensor],
                              generated_paths: List[str],
                              original_paths: List[str]) -> Dict[str, Dict[str, float]]:
        """여러 이미지 쌍에 대한 유사도를 한 번에 계산
        
        Args:
            generated_features (List[torch.Tensor]): 생성된 이미지들의 feature map 리스트
            original_features (List[torch.Tensor]): 원본 이미지들의 feature map 리스트
            generated_paths (List[str]): 생성된 이미지들의 경로 리스트
            original_paths (List[str]): 원본 이미지들의 경로 리스트
            
        Returns:
            Dict[str, Dict[str, float]]: 각 이미지 쌍의 유사도 (dictionary)
        """
        results = {}
        
        for gen_idx, gen_feature in enumerate(generated_features):
            gen_name = Path(generated_paths[gen_idx]).name
            results[gen_name] = {}
            
            for orig_idx, orig_feature in enumerate(original_features):
                orig_name = Path(original_paths[orig_idx]).name
                similarity = self.compute_cosine_similarity_full(
                    gen_feature, orig_feature
                )
                results[gen_name][orig_name] = similarity.item()
                
        return results
    
    def save_similarity_results(self,
                           results: Dict[str, Dict[str, float]], 
                           save_path: str,
                           include_stats: bool = True):
        """유사도 계산 결과 파일로 저장
        
        Args:
            results (Dict[str, Dict[str, float]]): 유사도 계산 결과
            save_path (str): 저장할 파일 경로
            include_stats (bool): 통계 정보 포함 여부
        """
        with open(save_path, 'w') as f:
            f.write("=== Image Similarity Results ===\n\n")
            
            for gen_img, similarities in results.items():
                f.write(f"Generated Image: {gen_img}\n")
                
                # 각 원본 이미지와의 유사도
                for orig_img, sim_score in similarities.items():
                    f.write(f"  - {orig_img}: {sim_score:.4f}\n")
                
                # 통계 정보 추가
                if include_stats:
                    sim_values = list(similarities.values())
                    f.write(f"  Statistics:\n")
                    f.write(f"    - Mean: {np.mean(sim_values):.4f}\n")
                    f.write(f"    - Std: {np.std(sim_values):.4f}\n")
                    f.write(f"    - Min: {np.min(sim_values):.4f}\n")
                    f.write(f"    - Max: {np.max(sim_values):.4f}\n")
                f.write("\n")

def main():
    # TestSamEncoder 인스턴스 생성
    encoder = TestSamEncoder(gpu_id=3)
    calculator = SamSimilarityCalculator()
    
    # 데이터 경로
    base_path = Path("/home/minelab/바탕화면/ANN/himeow/assets")
    generated_paths = [
        base_path / "encoder_test_dataset" / "generated1.jpg",
        base_path / "encoder_test_dataset" / "generated2.jpg"
    ]
    original_paths = [
        base_path / "encoder_test_dataset" / "origin1.jpg",
        base_path / "encoder_test_dataset" / "origin2.jpg"
    ]
    
    # Feature maps 추출
    generated_features = []
    original_features = []
    
    print("Processing generated images...")
    for path in generated_paths:
        outputs = encoder.process_image(str(path))
        generated_features.append(outputs.last_hidden_state)
        
    print("\nProcessing original images...")
    for path in original_paths:
        outputs = encoder.process_image(str(path))
        original_features.append(outputs.last_hidden_state)
    
    # 유사도 계산
    print("\nComputing similarities...")
    results = calculator.compute_batch_similarities(
        generated_features,
        original_features,
        [str(p) for p in generated_paths],
        [str(p) for p in original_paths]
    )
    
    # 결과 저장
    save_path = base_path / "encoder_test_result" / "sim_with_featuremap_all" / "similarity_results1.txt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    calculator.save_similarity_results(
        results,
        str(save_path),
        include_stats=True
    )
    
    print(f"\nResults saved to {save_path}")

if __name__ == "__main__":
    main()