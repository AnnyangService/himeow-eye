import numpy as np
from typing import Dict, List, Tuple, Set
import os
import json
from scipy.spatial.distance import cosine

class VectorMatcher:
    def __init__(self, 
                 origin_vector_path: str,
                 generated_vector_path: str,
                 k: int = 2,
                 similarity_threshold: float = 0.7):  # 유사도 스레시홀드 추가
        """
        Args:
            origin_vector_path: 원본 이미지 벡터 경로 (.npy)
            generated_vector_path: 생성된 이미지 벡터 경로 (.npy)
            k: 각 원본 이미지 하나당 매칭할 생성 이미지 수
            similarity_threshold: 최소 유사도 기준값 (이 값 이하의 유사도는 매칭하지 않음)
        """
        self.k = k
        self.similarity_threshold = similarity_threshold
        self.origin_vectors = np.load(origin_vector_path, allow_pickle=True).item()
        self.generated_vectors = np.load(generated_vector_path, allow_pickle=True).item()
        
        print(f"Loaded {len(self.origin_vectors)} original vectors and {len(self.generated_vectors)} generated vectors")
        print(f"Similarity threshold: {similarity_threshold}")

    def calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """두 벡터 간의 코사인 유사도 계산"""
        return 1 - cosine(vec1, vec2)

    def match_vectors(self, save_dir: str) -> Tuple[Dict[str, List[Tuple[str, float]]], Set[str]]:
        """벡터 매칭 실행"""
        os.makedirs(save_dir, exist_ok=True)
        results = {}
        matched_generated = set()  # 이미 매칭된 생성 이미지 추적
        
        # 각 원본 이미지에 대해
        for i, (orig_path, orig_vector) in enumerate(self.origin_vectors.items(), 1):
            print(f"\rProcessing original vector {i}/{len(self.origin_vectors)}", end='', flush=True)
            
            # 모든 생성 이미지와의 유사도 계산
            similarities = []
            for gen_path, gen_vector in self.generated_vectors.items():
                if gen_path not in matched_generated:  # 아직 매칭되지 않은 생성 이미지만 고려
                    similarity = self.calculate_similarity(orig_vector, gen_vector)
                    if similarity >= self.similarity_threshold:  # 스레시홀드 이상인 경우만 추가
                        similarities.append((gen_path, similarity))
            
            # 유사도 기준 내림차순 정렬
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 상위 k개 선택 (유사도가 threshold 이상인 것들 중에서)
            top_k_matches = similarities[:self.k]
            if top_k_matches:  # 매칭된 것이 있는 경우만 저장
                results[orig_path] = top_k_matches
                # 매칭된 생성 이미지 기록
                for gen_path, _ in top_k_matches:
                    matched_generated.add(gen_path)
        
        # 매칭되지 않은 생성 이미지 찾기
        unmatched_generated = set(self.generated_vectors.keys()) - matched_generated
        
        print("\nMatching completed!")
        
        # 결과 저장
        self.save_results(results, unmatched_generated, save_dir)
        
        return results, unmatched_generated

    def save_results(self, 
                    results: Dict[str, List[Tuple[str, float]]], 
                    unmatched: Set[str],
                    save_dir: str):
        """매칭 결과 저장"""
        # 1. 텍스트 형식
        with open(os.path.join(save_dir, 'matching_results.txt'), 'w') as f:
            f.write(f"Matching Results (Similarity Threshold: {self.similarity_threshold})\n")
            f.write("=" * 50 + "\n\n")
            
            # 매칭된 결과
            f.write("Matched Results\n")
            f.write("-" * 50 + "\n\n")
            for orig_path, matches in results.items():
                f.write(f"Original: {orig_path}\n")
                for gen_path, similarity in matches:
                    f.write(f"→ Generated: {gen_path} (similarity: {similarity:.4f})\n")
                f.write("\n")
            
            # 매칭되지 않은 원본 이미지
            unmatched_originals = set(self.origin_vectors.keys()) - set(results.keys())
            if unmatched_originals:
                f.write("\nUnmatched Original Images (No matches above threshold)\n")
                f.write("-" * 50 + "\n")
                for orig_path in sorted(unmatched_originals):
                    f.write(f"• {orig_path}\n")
            
            # 매칭되지 않은 생성 이미지
            f.write("\nUnmatched Generated Images\n")
            f.write("-" * 50 + "\n")
            for gen_path in sorted(unmatched):
                f.write(f"• {gen_path}\n")
        
        # 2. JSON 형식
        json_results = {
            "threshold": self.similarity_threshold,
            "matched": {
                orig_path: [
                    {"generated_path": gen_path, "similarity": float(sim)}
                    for gen_path, sim in matches
                ]
                for orig_path, matches in results.items()
            },
            "unmatched_originals": list(set(self.origin_vectors.keys()) - set(results.keys())),
            "unmatched_generated": list(unmatched)
        }
        
        with open(os.path.join(save_dir, 'matching_results.json'), 'w') as f:
            json.dump(json_results, f, indent=4)
        
        # 3. 매칭 통계 저장
        stats = self.get_matching_stats(results, unmatched)
        with open(os.path.join(save_dir, 'matching_stats.txt'), 'w') as f:
            f.write(f"Matching Statistics (Similarity Threshold: {self.similarity_threshold})\n")
            f.write("=" * 50 + "\n\n")
            for stat_name, stat_value in stats.items():
                f.write(f"{stat_name}: {stat_value}\n")

    def get_matching_stats(self, 
                          results: Dict[str, List[Tuple[str, float]]], 
                          unmatched: Set[str]) -> Dict:
        """매칭 통계 계산"""
        all_similarities = [sim for matches in results.values() 
                          for _, sim in matches]
        matched_generated = set(gen_path for matches in results.values() 
                              for gen_path, _ in matches)
        unmatched_originals = set(self.origin_vectors.keys()) - set(results.keys())
        
        stats = {
            'total_originals': len(self.origin_vectors),
            'matched_originals': len(results),
            'unmatched_originals': len(unmatched_originals),
            'total_generated': len(self.generated_vectors),
            'matched_generated': len(matched_generated),
            'unmatched_generated': len(unmatched),
        }
        
        if all_similarities:  # 매칭된 결과가 있는 경우만 유사도 통계 추가
            stats.update({
                'average_similarity': f"{np.mean(all_similarities):.4f}",
                'min_similarity': f"{min(all_similarities):.4f}",
                'max_similarity': f"{max(all_similarities):.4f}",
                'std_similarity': f"{np.std(all_similarities):.4f}"
            })
        
        return stats


if __name__ == "__main__":
    # 경로 설정
    origin_vector_path = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/vectors/origin.npy"
    generated_vector_path = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/vectors/generated.npy"
    save_dir = "/home/minelab/desktop/ANN/jojun/himeow-eye/test/vector_match_test"

    # 매칭 실행
    matcher = VectorMatcher(
        origin_vector_path=origin_vector_path,
        generated_vector_path=generated_vector_path,
        k=2, 
        similarity_threshold=0.85 
    )
    
    results, unmatched = matcher.match_vectors(save_dir=save_dir)