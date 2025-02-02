import os
import shutil
from vector_matcher import VectorMatcher

def filter_images(
    generated_dir: str,
    output_dir: str,
    origin_vector_path: str,
    generated_vector_path: str,
    k: int = 2,
    similarity_threshold: float = 0.85
):
    # 매칭 수행
    matcher = VectorMatcher(
        origin_vector_path=origin_vector_path,
        generated_vector_path=generated_vector_path,
        k=k,
        similarity_threshold=similarity_threshold
    )
    
    result = matcher.find_matches()
    
    # 결과 로그 저장
    matcher.save_results(result, output_dir)
    
    # 필터링된 이미지 저장을 위한 디렉토리 생성
    filtered_dir = os.path.join(output_dir, 'filtered_images')
    os.makedirs(filtered_dir, exist_ok=True)
    
    # 필터링된 생성 이미지들 복사
    print("\nCopying filtered images...")
    matched_generated = set()
    for matches in result.matched_pairs.values():
        for gen_path, _ in matches:
            if gen_path not in matched_generated:  # 중복 복사 방지
                matched_generated.add(gen_path)
                shutil.copy2(
                    os.path.join(generated_dir, gen_path),
                    os.path.join(filtered_dir, os.path.basename(gen_path))
                )
    
    print(f"\nFiltering completed!")
    print(f"Filtered images saved to: {filtered_dir}")
    print(f"Total generated images: {len(result.matched_pairs)}")
    print(f"Filtered generated images: {result.statistics['matched_generated']}")


if __name__ == "__main__":
    # 경로 설정
    generated_dir = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/generated"
    output_dir = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/filtered"
    origin_vector_path = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/vectors/origin.npy"
    generated_vector_path = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/vectors/generated.npy"

    # 필터링 실행
    filter_images(
        generated_dir=generated_dir,
        output_dir=output_dir,
        origin_vector_path=origin_vector_path,
        generated_vector_path=generated_vector_path,
        k=2,
        similarity_threshold=0.85
    )