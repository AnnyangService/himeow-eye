import os
import json
from collections import defaultdict

def count_breeds_in_directory(directory):
    # 찾고자 하는 품종 리스트
    target_breeds = [
        "페르시안",
        "엑조틱 숏헤어",
        "히말라얀",
        "스코티시 폴드",
        "브리티시 숏헤어",
        "셀커크 렉스"
    ]
    
    # 각 품종별 카운트를 저장할 딕셔너리 초기화
    breed_counts = defaultdict(int)
    
    # 디렉토리 내의 모든 JSON 파일을 순회
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # breed 정보 추출
                    breed = data.get('images', {}).get('meta', {}).get('breed')
                    
                    # 타겟 품종에 해당하면 카운트 증가
                    if breed in target_breeds:
                        breed_counts[breed] += 1
                        
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    
    return breed_counts

# 각 폴더별로 실행
folders = ['train/normal', 'train/abnormal', 'validation/normal', 'validation/abnormal']
base_path = '/home/minelab/바탕화면/ANN/himeow/origin_datasets/'

for folder in folders:
    full_path = os.path.join(base_path, folder)
    print(f"\n=== {folder} 폴더 결과 ===")
    counts = count_breeds_in_directory(full_path)
    
    # 결과 출력
    for breed in ["페르시안", "엑조틱 숏헤어", "히말라얀", "스코티시 폴드", "브리티시 숏헤어", "셀커크 렉스"]:
        print(f"{breed}: {counts[breed]}개")