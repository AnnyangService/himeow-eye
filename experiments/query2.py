import sys
import os
sys.path.append('/home/minelab/desktop/ANN/jojun/himeow-eye')

from models.encoder.filtering.extract import FeatureExtractor
from models.encoder.filtering.channel_selection.select import ChannelSelector
from models.encoder.filtering.filter2 import filter_images
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
generated_dir = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/generated"
output_dir = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/filtered"
origin_vector_path = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/vectors/origin.npy"
generated_vector_path = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/vectors/generated.npy"

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
filter_images(
    generated_dir=generated_dir,
    output_dir=output_dir,
    origin_vector_path=origin_vector_path,
    generated_vector_path=generated_vector_path,
    k=2,
    similarity_min=0.5,  # 최소 유사도
    similarity_max=0.8   # 최대 유사도
)

print("\n=== Processing Completed ===")