import os
import shutil
from pathlib import Path

def collect_and_clean_dataset(source_dir, target_dir):
    # 대상 디렉토리 생성
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    # 이미지 파일 확장자
    image_extensions = ('.jpg', '.jpeg', '.png')
    
    # 모든 이미지 파일 수집 및 복사
    total_copied = 0
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_dir, file)
                shutil.copy2(source_path, target_path)
                total_copied += 1
    
    # .json 파일 제거
    json_removed = 0
    for file in os.listdir(target_dir):
        if file.lower().endswith('.json'):
            os.remove(os.path.join(target_dir, file))
            json_removed += 1
    
    return total_copied, json_removed

if __name__ == "__main__":
    # 경로 설정
    source_directory = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/filtered_by_breeds_datasets/brachy"
    target_directory = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/keratitis"
    
    # 실행
    copied, removed = collect_and_clean_dataset(source_directory, target_directory)
    
    # 결과 출력
    print(f"복사된 이미지 파일 수: {copied}")
    print(f"제거된 JSON 파일 수: {removed}")
    print(f"최종 이미지 파일 수: {len([f for f in os.listdir(target_directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])}")