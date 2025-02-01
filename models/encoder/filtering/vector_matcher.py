import numpy as np
from typing import Dict, List, Tuple
import os
import json
from scipy.spatial.distance import cosine
import glob
import sys

sys.path.append('/home/minelab/desktop/ANN/jojun/himeow-eye')
from models.encoder.filtering.extract import FeatureExtractor
from models.encoder.filtering.channel_selection.select import ChannelSelector

class VectorMatcher:
    def __init__(self, 
                 origin_vector_path: str,
                 encoder_config: dict,
                 k: int = 2):
        """
        Args:
            origin_vector_path: 원본 이미지 벡터 경로
            encoder_config: 인코더 설정
            k: 각 원본 이미지 하나당 k개의 생성 이미지를 매칭함
        """
        self.k = k
        self.origin_vectors = np.load(origin_vector_path, allow_pickle=True).item()
        
        # 특징 추출기와 채널 선택기 초기화
        self.extractor = FeatureExtractor(
            checkpoint_path=encoder_config.get('checkpoint_path'),
            gpu_id=encoder_config.get('gpu_id', 3)
        )
        self.selector = ChannelSelector(
            padding_config=encoder_config.get('padding_config'),
            scoring_config=encoder_config.get('scoring_config')
        )
        
        print(f"Loaded {len(self.origin_vectors)} original vectors")

    def calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """두 벡터 간의 코사인 유사도 계산"""
        return 1 - cosine(vec1, vec2)

    def extract_and_select_features(self, image_path: str) -> np.ndarray:
        """이미지에서 특징을 추출하고 중요 채널 선택"""
        try:
            # 1. 특징 추출
            features = self.extractor.extract_features(image_path)
            if features is None:  # 추출 실패 시
                raise ValueError(f"Failed to extract features from {image_path}")
            features = features.contiguous().clone()
            
            # 2. 채널 선택
            selected_features, _ = self.selector.select_channels(features)
            return selected_features[0].cpu().numpy().copy()
            
        except Exception as e:
            print(f"\nWarning: Error in feature extraction for {image_path}: {str(e)}")
            return None

    def process_generated_images(self, 
                               generated_image_dir: str,
                               save_dir: str) -> Dict[str, Dict[int, List[Tuple[int, float]]]]:
        """생성 이미지들을 처리 및 매칭"""
        os.makedirs(save_dir, exist_ok=True)
        results = {}
        
        # 이미지 파일 찾기
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(generated_image_dir, '**', ext), recursive=True))
        
        # 각 이미지 처리
        for i, image_path in enumerate(image_files, 1):
            try:
                print(f"\rProcessing image {i}/{len(image_files)}: {os.path.basename(image_path)}", 
                      end='', flush=True)
                
                # 특징 벡터 추출 및 선택
                gen_vector = self.extract_and_select_features(image_path)
                if gen_vector is None:  # 추출 실패 시
                    print(f"\nSkipping {image_path} due to feature extraction error")
                    continue
                
                # 매칭 찾기
                matches = {}
                used_generated = set()
                
                # 각 원본 벡터에 대해 유사도 계산
                for orig_idx, (_, orig_vector) in enumerate(self.origin_vectors.items()):
                    matches[orig_idx] = []
                    
                    # 유사도 계산
                    similarity = self.calculate_similarity(gen_vector, orig_vector)
                    similarities = [(orig_idx, similarity)]
                    
                    # 정렬 및 매칭
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    
                    # 상위 k개 선택
                    for j in range(min(self.k, len(similarities))):
                        orig_idx, similarity = similarities[j]
                        if orig_idx not in used_generated:  # 중복 방지
                            matches[orig_idx].append((i-1, similarity))  # i-1은 현재 생성 이미지의 인덱스
                            used_generated.add(orig_idx)
                
                # 결과 저장
                rel_path = os.path.relpath(image_path, generated_image_dir)
                results[rel_path] = matches
                
                # 개별 결과 파일 저장
                self.save_individual_result(
                    matches, 
                    save_dir, 
                    os.path.splitext(os.path.basename(image_path))[0]
                )
                
            except Exception as e:
                print(f"\nError processing {image_path}: {str(e)}")
        
        print("\nProcessing completed!")
        
        # 전체 결과 저장
        self.save_all_results(results, save_dir)
        
        return results

    def save_individual_result(self, 
                             matches: Dict[int, List[Tuple[int, float]]], 
                             save_dir: str,
                             base_name: str):
        """개별 이미지의 매칭 결과 저장"""
        result_path = os.path.join(save_dir, f"{base_name}_matches.txt")
        with open(result_path, 'w') as f:
            f.write(f"Matching Results for {base_name}\n")
            f.write("=" * 50 + "\n\n")
            
            for orig_idx, matched_list in matches.items():
                matched_str = ", ".join([f"생성{gen_idx}(유사도:{sim:.4f})" 
                                       for gen_idx, sim in matched_list])
                f.write(f"원본 {orig_idx} -> {matched_str}\n")
            
            # 통계 추가
            stats = self.get_matching_stats(matches)
            f.write("\nMatching Statistics\n")
            f.write("=" * 50 + "\n")
            for stat_name, stat_value in stats.items():
                f.write(f"{stat_name}: {stat_value:.4f}\n")

    def save_all_results(self, 
                        results: Dict[str, Dict[int, List[Tuple[int, float]]]], 
                        save_dir: str):
        """전체 매칭 결과 저장"""
        # 1. 텍스트 형식
        with open(os.path.join(save_dir, 'all_matches.txt'), 'w') as f:
            f.write("All Matching Results\n")
            f.write("=" * 50 + "\n\n")
            
            for image_path, matches in results.items():
                f.write(f"\nResults for {image_path}:\n")
                f.write("-" * 50 + "\n")
                for orig_idx, matched_list in matches.items():
                    matched_str = ", ".join([f"생성{gen_idx}(유사도:{sim:.4f})" 
                                           for gen_idx, sim in matched_list])
                    f.write(f"원본 {orig_idx} -> {matched_str}\n")
                f.write("=" * 50 + "\n")
        
        # 2. JSON 형식
        json_results = {
            image_path: {
                str(orig_idx): [(int(gen_idx), float(sim)) 
                               for gen_idx, sim in matched_list]
                for orig_idx, matched_list in matches.items()
            }
            for image_path, matches in results.items()
        }
        
        with open(os.path.join(save_dir, 'all_matches.json'), 'w') as f:
            json.dump(json_results, f, indent=4)

    def get_matching_stats(self, matches: Dict[int, List[Tuple[int, float]]]) -> Dict:
        """매칭 통계 계산"""
        match_counts = [len(m) for m in matches.values()]
        used_generated = set()
        for matched_list in matches.values():
            used_generated.update(gen_idx for gen_idx, _ in matched_list)
        
        all_similarities = [sim for matched_list in matches.values() 
                          for _, sim in matched_list]
        
        return {
            'total_matches': sum(match_counts),
            'avg_matches_per_original': np.mean(match_counts),
            'total_unique_generated': len(used_generated),
            'avg_similarity': np.mean(all_similarities) if all_similarities else 0,
            'min_similarity': min(all_similarities) if all_similarities else 0,
            'max_similarity': max(all_similarities) if all_similarities else 0
        }

if __name__ == "__main__":
    # 설정
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
    origin_vector_path = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/vectors/origin.npy"
    generated_image_dir = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/generated"
    save_dir = "/home/minelab/desktop/ANN/jojun/himeow-eye/test/vector_match_test"

    # 매칭 실행
    matcher = VectorMatcher(
        origin_vector_path=origin_vector_path,
        encoder_config=config,
        k=3
    )
    
    results = matcher.process_generated_images(
        generated_image_dir=generated_image_dir,
        save_dir=save_dir
    )