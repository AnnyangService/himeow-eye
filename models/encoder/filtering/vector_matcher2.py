from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
import numpy as np
import os
import json
from scipy.spatial.distance import cosine

@dataclass
class MatchingResult:
    """매칭 결과를 담는 데이터 클래스"""
    matched_pairs: Dict[str, List[Tuple[str, float]]]
    unmatched_originals: Set[str]
    unmatched_generated: Set[str]
    statistics: Dict[str, float]

class VectorMatcher:
    def __init__(self, 
                 origin_vector_path: str,
                 generated_vector_path: str,
                 k: int = 2,
                 similarity_min: float = 0.7,
                 similarity_max: float = 1.0):
        """
        Args:
            origin_vector_path: 원본 이미지 벡터 경로 (.npy)
            generated_vector_path: 생성된 이미지 벡터 경로 (.npy)
            k: 각 원본 이미지 하나당 매칭할 생성 이미지 수
            similarity_min: 최소 유사도 기준값
            similarity_max: 최대 유사도 기준값
        """
        self.k = k
        self.similarity_min = similarity_min
        self.similarity_max = similarity_max
        self.origin_vectors = np.load(origin_vector_path, allow_pickle=True).item()
        self.generated_vectors = np.load(generated_vector_path, allow_pickle=True).item()
        
        print(f"Loaded {len(self.origin_vectors)} original vectors and {len(self.generated_vectors)} generated vectors")
        print(f"Similarity range: {similarity_min} - {similarity_max}")

    def calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """두 벡터 간의 코사인 유사도 계산"""
        return 1 - cosine(vec1, vec2)

    def find_matches(self) -> MatchingResult:
        """벡터 매칭 실행"""
        results = {}
        matched_generated = set()
        
        for i, (orig_path, orig_vector) in enumerate(self.origin_vectors.items(), 1):
            print(f"\rProcessing original vector {i}/{len(self.origin_vectors)}", end='', flush=True)
            
            similarities = []
            for gen_path, gen_vector in self.generated_vectors.items():
                if gen_path not in matched_generated:
                    similarity = self.calculate_similarity(orig_vector, gen_vector)
                    # 유사도가 지정된 범위 내에 있는 경우만 추가
                    if self.similarity_min <= similarity <= self.similarity_max:
                        similarities.append((gen_path, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_k_matches = similarities[:self.k]
            
            if top_k_matches:
                results[orig_path] = top_k_matches
                for gen_path, _ in top_k_matches:
                    matched_generated.add(gen_path)
        
        unmatched_originals = set(self.origin_vectors.keys()) - set(results.keys())
        unmatched_generated = set(self.generated_vectors.keys()) - matched_generated
        
        stats = self._calculate_statistics(results, unmatched_originals, unmatched_generated)
        
        print("\nMatching completed!")
        
        return MatchingResult(
            matched_pairs=results,
            unmatched_originals=unmatched_originals,
            unmatched_generated=unmatched_generated,
            statistics=stats
        )

    def _calculate_statistics(self, 
                            results: Dict[str, List[Tuple[str, float]]], 
                            unmatched_originals: Set[str],
                            unmatched_generated: Set[str]) -> Dict[str, float]:
        """매칭 통계 계산"""
        all_similarities = [sim for matches in results.values() 
                          for _, sim in matches]
        matched_generated = set(gen_path for matches in results.values() 
                              for gen_path, _ in matches)
        
        stats = {
            'total_originals': len(self.origin_vectors),
            'matched_originals': len(results),
            'unmatched_originals': len(unmatched_originals),
            'total_generated': len(self.generated_vectors),
            'matched_generated': len(matched_generated),
            'unmatched_generated': len(unmatched_generated),
        }
        
        if all_similarities:
            stats.update({
                'average_similarity': float(np.mean(all_similarities)),
                'min_similarity': float(min(all_similarities)),
                'max_similarity': float(max(all_similarities)),
                'std_similarity': float(np.std(all_similarities))
            })
        
        return stats

    def save_results(self, result: MatchingResult, save_dir: str):
        """매칭 결과 저장"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 텍스트 형식
        with open(os.path.join(save_dir, 'matching_results.txt'), 'w') as f:
            f.write(f"Matching Results (Similarity Range: {self.similarity_min} - {self.similarity_max})\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Matched Results\n")
            f.write("-" * 50 + "\n\n")
            for orig_path, matches in result.matched_pairs.items():
                f.write(f"Original: {orig_path}\n")
                for gen_path, similarity in matches:
                    f.write(f"→ Generated: {gen_path} (similarity: {similarity:.4f})\n")
                f.write("\n")
            
            if result.unmatched_originals:
                f.write("\nUnmatched Original Images\n")
                f.write("-" * 50 + "\n")
                for orig_path in sorted(result.unmatched_originals):
                    f.write(f"• {orig_path}\n")
            
            f.write("\nUnmatched Generated Images\n")
            f.write("-" * 50 + "\n")
            for gen_path in sorted(result.unmatched_generated):
                f.write(f"• {gen_path}\n")
        
        # 2. JSON 형식
        json_results = {
            "similarity_range": {
                "min": self.similarity_min,
                "max": self.similarity_max
            },
            "matched": {
                orig_path: [
                    {"generated_path": gen_path, "similarity": float(sim)}
                    for gen_path, sim in matches
                ]
                for orig_path, matches in result.matched_pairs.items()
            },
            "unmatched_originals": list(result.unmatched_originals),
            "unmatched_generated": list(result.unmatched_generated),
            "statistics": result.statistics
        }
        
        with open(os.path.join(save_dir, 'matching_results.json'), 'w') as f:
            json.dump(json_results, f, indent=4)
        
        # 3. 매칭 통계 저장
        with open(os.path.join(save_dir, 'matching_stats.txt'), 'w') as f:
            f.write(f"Matching Statistics (Similarity Range: {self.similarity_min} - {self.similarity_max})\n")
            f.write("=" * 50 + "\n\n")
            for stat_name, stat_value in result.statistics.items():
                f.write(f"{stat_name}: {stat_value}\n")


if __name__ == "__main__":
    # 경로 설정
    origin_vector_path = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/vectors/origin.npy"
    generated_vector_path = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/vectors/generated.npy"
    save_dir = "/home/minelab/desktop/ANN/jojun/himeow-eye/test/vector_match_test2"

    # 매칭 실행
    matcher = VectorMatcher(
        origin_vector_path=origin_vector_path,
        generated_vector_path=generated_vector_path,
        k=2,
        similarity_min=0.7,  # 최소 유사도
        similarity_max=0.85   # 최대 유사도
    )
    
    result = matcher.find_matches()
    matcher.save_results(result, save_dir)