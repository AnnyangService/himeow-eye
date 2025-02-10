import os
import shutil
from .vector_matcher2 import VectorMatcher

def filter_images(
    generated_dir: str,
    output_dir: str,
    origin_vector_path: str,
    generated_vector_path: str,
    k: int = 2,
    similarity_min: float = 0.7,
    similarity_max: float = 0.85
):
    """
    생성된 이미지들을 유사도 범위에 따라 필터링하여 저장
    
    Args:
        generated_dir: 생성된 이미지들이 있는 디렉토리 경로
        output_dir: 결과를 저장할 디렉토리 경로
        origin_vector_path: 원본 이미지 벡터 파일 경로
        generated_vector_path: 생성된 이미지 벡터 파일 경로
        k: 각 원본 이미지당 매칭할 생성 이미지 수
        similarity_min: 최소 유사도 기준값
        similarity_max: 최대 유사도 기준값
    """
    # 매칭 수행
    matcher = VectorMatcher(
        origin_vector_path=origin_vector_path,
        generated_vector_path=generated_vector_path,
        k=k,
        similarity_min=similarity_min,
        similarity_max=similarity_max
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
        for gen_path, similarity in matches:
            if gen_path not in matched_generated:  # 중복 복사 방지
                matched_generated.add(gen_path)
                # 유사도 값을 파일 이름에 포함
                base_name, ext = os.path.splitext(os.path.basename(gen_path))
                new_name = f"{base_name}_sim{similarity:.4f}{ext}"
                shutil.copy2(
                    os.path.join(generated_dir, gen_path),
                    os.path.join(filtered_dir, new_name)
                )
    
    print(f"\nFiltering completed!")
    print(f"Filtered images saved to: {filtered_dir}")
    print(f"Total original images: {result.statistics['total_originals']}")
    print(f"Matched original images: {result.statistics['matched_originals']}")
    print(f"Total generated images: {result.statistics['total_generated']}")
    print(f"Filtered generated images: {result.statistics['matched_generated']}")
    print(f"Average similarity: {result.statistics.get('average_similarity', 0):.4f}")
    print(f"Similarity range: {similarity_min:.2f} - {similarity_max:.2f}")


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
        similarity_min=0.7, 
        similarity_max=0.85
    )