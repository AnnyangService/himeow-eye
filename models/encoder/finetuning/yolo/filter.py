import cv2
import numpy as np
import shutil
from pathlib import Path
import csv
from datetime import datetime

def extract_blue_mask(img):
    """
    파란색 마스크를 추출하는 함수
    HSV 색공간에서 파란색 영역을 추출
    """
    # BGR to HSV 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 파란색 범위 정의 (약간 넓게 잡아서 다양한 파란 색조를 포함)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    
    # 마스크 생성
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # 노이즈 제거
    kernel = np.ones((3,3), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    
    return blue_mask

def evaluate_mask_quality(mask):
    """
    마스크의 품질을 평가하는 함수
    반환값: 마스크 면적 비율, 가장 큰 연결요소의 비율, 전체 마스크 크기
    """
    # 마스크가 비어있는 경우 체크
    if np.sum(mask) == 0:
        return 0, 0, 0
        
    # 마스크 영역 비율 계산
    mask_area_ratio = np.sum(mask > 0) / mask.size
    
    # 연결된 컴포넌트 찾기
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    
    # 배경(label 0)을 제외한 컴포넌트들의 크기
    component_sizes = stats[1:, cv2.CC_STAT_AREA]
    
    if len(component_sizes) > 0:
        # 가장 큰 컴포넌트의 비율 계산
        largest_component_ratio = np.max(component_sizes) / np.sum(mask > 0)
    else:
        largest_component_ratio = 0
    
    return mask_area_ratio, largest_component_ratio, np.sum(mask > 0)

def filter_predictions(
    pred_dir,
    output_dir,
    good_output_dir="good_predictions",
    poor_output_dir="poor_predictions",
    max_area_threshold=0.8,
    min_area_threshold=0.01,
    min_component_ratio=0.85
):
    # 출력 디렉토리 설정
    output_base = Path(output_dir)
    good_dir = output_base / good_output_dir
    poor_dir = output_base / poor_output_dir
    poor_large_dir = poor_dir / "too_large"
    poor_small_dir = poor_dir / "too_small"
    poor_split_dir = poor_dir / "split_regions"
    poor_empty_dir = poor_dir / "no_mask"
    
    # 디렉토리 생성
    for dir_path in [good_dir, poor_large_dir, poor_small_dir, poor_split_dir, poor_empty_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    pred_dir = Path(pred_dir)
    
    # 로그 파일 설정
    log_file = output_base / "filtering_log.csv"
    log_header = ["timestamp", "filename", "result", "area_ratio", "largest_component_ratio", "total_area", "destination"]
    
    # CSV 파일 작성 - 한 번만 열기
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(log_header)
        
        # 모든 예측 결과 처리
        for pred_path in pred_dir.glob("*"):
            if pred_path.is_file() and pred_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # 이미지 읽기
                img = cv2.imread(str(pred_path))
                if img is None:
                    print(f"Failed to read image: {pred_path.name}")
                    continue
                
                # 파란색 마스크 추출
                blue_mask = extract_blue_mask(img)
                
                # 마스크 품질 평가
                area_ratio, largest_component_ratio, total_area = evaluate_mask_quality(blue_mask)
                
                # 결과 저장 위치 결정
                if area_ratio == 0:
                    dest_dir = poor_empty_dir
                    reason = "No mask detected"
                elif area_ratio > max_area_threshold:
                    dest_dir = poor_large_dir
                    reason = f"Mask too large: {area_ratio:.3f}"
                elif area_ratio < min_area_threshold:
                    dest_dir = poor_small_dir
                    reason = f"Mask too small: {area_ratio:.3f}"
                elif largest_component_ratio < min_component_ratio:
                    dest_dir = poor_split_dir
                    reason = f"Multiple separated regions detected: {largest_component_ratio:.3f}"
                else:
                    dest_dir = good_dir
                    reason = "Good quality mask"
                
                # 로그 기록
                log_entry = [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    pred_path.name,
                    reason,
                    f"{area_ratio:.3f}",
                    f"{largest_component_ratio:.3f}",
                    total_area,
                    dest_dir.name
                ]
                
                writer.writerow(log_entry)
                
                # 결과 저장
                shutil.copy2(pred_path, dest_dir / pred_path.name)
                
                print(f"Processed {pred_path.name}")

if __name__ == "__main__":
    filter_predictions(
        pred_dir="/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/other_diseases_gtmasks_origin",
        output_dir="/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/other_diseases_gtmasks",
        max_area_threshold=0.8,
        min_area_threshold=0.01,
        min_component_ratio=0.85
    )