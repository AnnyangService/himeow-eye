import sys
import os
import shutil
sys.path.append('/home/minelab/desktop/ANN/jojun/himeow-eye')

from models.encoder.filtering.extract import FeatureExtractor
from models.encoder.filtering.channel_selection.select import ChannelSelector
from models.encoder.filtering.vector_filter import VectorFilter  
from models.encoder.filtering.vectorization import vectorize_directory

# 벡터화 설정
config = {
    'checkpoint_path': "/home/minelab/desktop/ANN/jojun/himeow-eye/models/encoder/finetuning/custom_models/best_checkpoint.pth",
    'gpu_id': 3,
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
origin_dir = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/keratitis"
generated_dir = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/keratitis_generated"
output_dir = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/filtered"
origin_vector_path = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/vectors/origin.npy"
generated_vector_path = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/vectors/generated.npy"

# 벡터 저장 디렉토리 생성
os.makedirs(os.path.dirname(origin_vector_path), exist_ok=True)
os.makedirs(os.path.dirname(generated_vector_path), exist_ok=True)

# 벡터 파일 존재 여부 확인
origin_exists = os.path.exists(origin_vector_path)
generated_exists = os.path.exists(generated_vector_path)

# 1. 벡터화
if not origin_exists:
    print("\nVectorizing origin images...")
    origin_vectors = vectorize_directory(
        image_dir=origin_dir,
        save_path=origin_vector_path,
        **config
    )
else:
    print("\nOrigin vector file already exists, skipping origin vectorization...")

if not generated_exists:
    print("\nVectorizing generated images...")
    generated_vectors = vectorize_directory(
        image_dir=generated_dir,
        save_path=generated_vector_path,
        **config
    )
else:
    print("\nGenerated vector file already exists, skipping generated vectorization...")

# 2. 필터링
print("\n=== Starting Filtering ===")

# VectorFilter 인스턴스 생성
vector_filter = VectorFilter(
    origin_vector_path=origin_vector_path,
    generated_vector_path=generated_vector_path
)

# 필터링 실행
result = vector_filter.filter_vectors()

# 결과 저장
vector_filter.save_results(result, output_dir)

# 이미지 파일 복사
filtered_dir = os.path.join(output_dir, 'filtered_images')
rejected_dir = os.path.join(output_dir, 'rejected_images')
os.makedirs(filtered_dir, exist_ok=True)
os.makedirs(rejected_dir, exist_ok=True)

# 필터링된 이미지 복사
print("\nCopying filtered images...")
for gen_path, similarity in result.filtered_images.items():
    base_name, ext = os.path.splitext(os.path.basename(gen_path))
    new_name = f"{base_name}_sim{similarity:.4f}{ext}"
    shutil.copy2(
        os.path.join(generated_dir, gen_path),
        os.path.join(filtered_dir, new_name)
    )

# 거부된 이미지 복사
print("Copying rejected images...")
for gen_path in result.rejected_images:
    shutil.copy2(
        os.path.join(generated_dir, gen_path),
        os.path.join(rejected_dir, os.path.basename(gen_path))
    )

print("\n=== Processing Completed ===")
print(f"Filtered images saved to: {filtered_dir}")
print(f"Rejected images saved to: {rejected_dir}")
print(f"Total images: {result.statistics['total_generated']}")
print(f"Filtered images: {result.statistics['filtered_count']}")
print(f"Rejected images: {result.statistics['rejected_count']}")
print(f"Average similarity: {result.statistics['average_similarity']:.4f}")
print(f"Threshold: {result.statistics['threshold']:.4f}")