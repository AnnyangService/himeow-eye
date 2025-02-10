from dataclasses import dataclass
from typing import Dict, List, Set
import numpy as np
import os
import json
import shutil
from scipy.spatial.distance import cosine

@dataclass
class FilterResult:
    """필터링 결과를 담는 데이터 클래스"""
    filtered_images: Dict[str, float]  # {image_path: similarity_score}
    rejected_images: Set[str]
    statistics: Dict[str, float]

class VectorFilter:
    def __init__(self, 
                 origin_vector_path: str,
                 generated_vector_path: str):
        """
        Args:
            origin_vector_path: 원본 이미지 벡터 경로 (.npy)
            generated_vector_path: 생성된 이미지 벡터 경로 (.npy)
        """
        self.origin_vectors = np.load(origin_vector_path, allow_pickle=True).item()
        self.generated_vectors = np.load(generated_vector_path, allow_pickle=True).item()
        
        # 원본 벡터들의 평균 계산
        self.origin_mean = np.mean(list(self.origin_vectors.values()), axis=0)
        print(f"Loaded {len(self.origin_vectors)} original vectors and {len(self.generated_vectors)} generated vectors")

    def calculate_similarity(self, vec: np.ndarray) -> float:
        """벡터와 원본 평균 벡터 간의 코사인 유사도 계산"""
        return 1 - cosine(vec, self.origin_mean)

    def filter_vectors(self) -> FilterResult:
        """벡터 필터링 실행"""
        filtered_images = {}
        rejected_images = set()
        
        # 모든 생성된 이미지의 유사도 계산
        all_similarities = []
        for gen_path, gen_vector in self.generated_vectors.items():
            similarity = self.calculate_similarity(gen_vector)
            all_similarities.append(similarity)
        
        # 평균 유사도를 threshold로 사용
        threshold = np.mean(all_similarities)
        print(f"Similarity threshold: {threshold:.4f}")
        
        # threshold를 기준으로 필터링
        for gen_path, gen_vector in self.generated_vectors.items():
            similarity = self.calculate_similarity(gen_vector)
            if similarity >= threshold:
                filtered_images[gen_path] = similarity
            else:
                rejected_images.add(gen_path)
        
        stats = self._calculate_statistics(filtered_images, rejected_images, all_similarities)
        
        print("\nFiltering completed!")
        
        return FilterResult(
            filtered_images=filtered_images,
            rejected_images=rejected_images,
            statistics=stats
        )

    def _calculate_statistics(self, 
                            filtered_images: Dict[str, float],
                            rejected_images: Set[str],
                            all_similarities: List[float]) -> Dict[str, float]:
        """필터링 통계 계산"""
        stats = {
            'total_generated': len(self.generated_vectors),
            'filtered_count': len(filtered_images),
            'rejected_count': len(rejected_images),
            'average_similarity': float(np.mean(all_similarities)),
            'threshold': float(np.mean(all_similarities)),
            'min_similarity': float(min(all_similarities)),
            'max_similarity': float(max(all_similarities)),
            'std_similarity': float(np.std(all_similarities))
        }
        
        return stats

    def save_results(self, result: FilterResult, save_dir: str):
        """필터링 결과 저장"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 텍스트 형식
        with open(os.path.join(save_dir, 'filtering_results.txt'), 'w') as f:
            f.write(f"Filtering Results (Threshold: {result.statistics['threshold']:.4f})\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Filtered Images\n")
            f.write("-" * 50 + "\n")
            for gen_path, similarity in sorted(result.filtered_images.items(), key=lambda x: x[1], reverse=True):
                f.write(f"• {gen_path} (similarity: {similarity:.4f})\n")
            
            f.write("\nRejected Images\n")
            f.write("-" * 50 + "\n")
            for gen_path in sorted(result.rejected_images):
                f.write(f"• {gen_path}\n")
        
        # 2. JSON 형식
        json_results = {
            "threshold": float(result.statistics['threshold']),
            "filtered_images": {
                path: float(sim) for path, sim in result.filtered_images.items()
            },
            "rejected_images": list(result.rejected_images),
            "statistics": result.statistics
        }
        
        with open(os.path.join(save_dir, 'filtering_results.json'), 'w') as f:
            json.dump(json_results, f, indent=4)
        
        # 3. 필터링 통계 저장
        with open(os.path.join(save_dir, 'filtering_stats.txt'), 'w') as f:
            f.write(f"Filtering Statistics (Threshold: {result.statistics['threshold']:.4f})\n")
            f.write("=" * 50 + "\n\n")
            for stat_name, stat_value in result.statistics.items():
                f.write(f"{stat_name}: {stat_value}\n")


def filter_images(
    generated_dir: str,
    output_dir: str,
    origin_vector_path: str,
    generated_vector_path: str
):
    """
    생성된 이미지들을 평균 유사도 기준으로 필터링하여 저장
    
    Args:
        generated_dir: 생성된 이미지들이 있는 디렉토리 경로
        output_dir: 결과를 저장할 디렉토리 경로
        origin_vector_path: 원본 이미지 벡터 파일 경로
        generated_vector_path: 생성된 이미지 벡터 파일 경로
    """
    # 필터링 수행
    filter = VectorFilter(
        origin_vector_path=origin_vector_path,
        generated_vector_path=generated_vector_path
    )
    
    result = filter.filter_vectors()
    
    # 결과 로그 저장
    filter.save_results(result, output_dir)
    
    # 필터링된 이미지 저장을 위한 디렉토리 생성
    filtered_dir = os.path.join(output_dir, 'filtered_images')
    rejected_dir = os.path.join(output_dir, 'rejected_images')
    os.makedirs(filtered_dir, exist_ok=True)
    os.makedirs(rejected_dir, exist_ok=True)
    
    # 필터링된 이미지들 복사
    print("\nCopying filtered images...")
    for gen_path, similarity in result.filtered_images.items():
        base_name, ext = os.path.splitext(os.path.basename(gen_path))
        new_name = f"{base_name}_sim{similarity:.4f}{ext}"
        shutil.copy2(
            os.path.join(generated_dir, gen_path),
            os.path.join(filtered_dir, new_name)
        )
    
    # 거부된 이미지들 복사
    print("Copying rejected images...")
    for gen_path in result.rejected_images:
        shutil.copy2(
            os.path.join(generated_dir, gen_path),
            os.path.join(rejected_dir, os.path.basename(gen_path))
        )
    
    print(f"\nFiltering completed!")
    print(f"Filtered images saved to: {filtered_dir}")
    print(f"Rejected images saved to: {rejected_dir}")
    print(f"Total generated images: {result.statistics['total_generated']}")
    print(f"Filtered images: {result.statistics['filtered_count']}")
    print(f"Rejected images: {result.statistics['rejected_count']}")
    print(f"Average similarity: {result.statistics['average_similarity']:.4f}")
    print(f"Threshold: {result.statistics['threshold']:.4f}")


if __name__ == "__main__":
    # 경로 설정
    generated_dir = "/path/to/generated/images"
    output_dir = "/path/to/output/directory"
    origin_vector_path = "/path/to/origin.npy"
    generated_vector_path = "/path/to/generated.npy"

    # 필터링 실행
    filter_images(
        generated_dir=generated_dir,
        output_dir=output_dir,
        origin_vector_path=origin_vector_path,
        generated_vector_path=generated_vector_path
    )