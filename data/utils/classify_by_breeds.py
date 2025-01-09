import os
import json
import shutil
from collections import defaultdict

def classify_by_breeds(src_directory, dst_base):
    # 단두종 품종 리스트
    brachycephalic_breeds = [
        "페르시안", "히말라얀", "엑조틱 숏헤어", 
        "스코티시 폴드", "브리티시 숏헤어", "셀커크 렉스"
    ]
    
    # 목적지 디렉토리 생성
    for face_type in ["brachy", "non_brachy"]:
        for condition in ["normal", "abnormal"]:
            os.makedirs(os.path.join(dst_base, face_type, condition), exist_ok=True)
    
    # 카운터 초기화
    counts = {
        'brachycephalic': {'normal': 0, 'abnormal': 0},
        'non_brachycephalic': {'normal': 0, 'abnormal': 0}
    }
    
    # 디렉토리 내의 모든 JSON 파일을 순회
    for root, dirs, files in os.walk(src_directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # breed 정보 추출
                    breed = data.get('images', {}).get('meta', {}).get('breed')
                    
                    # normal/abnormal 구분 (경로에서 추출)
                    path_parts = root.lower().split(os.sep)
                    is_normal = any(part == 'normal' for part in path_parts) and not any(part == 'abnormal' for part in path_parts)
                    condition = 'normal' if is_normal else 'abnormal'
                    
                    # 목적지 폴더 결정
                    if breed in brachycephalic_breeds:
                        dst_dir = os.path.join(dst_base, "brachy", condition)
                        counts['brachycephalic'][condition] += 1
                    else:
                        dst_dir = os.path.join(dst_base, "non_brachy", condition)
                        counts['non_brachycephalic'][condition] += 1
                    
                    # JSON 파일 복사
                    shutil.copy2(file_path, os.path.join(dst_dir, file))
                    
                    # 관련된 이미지 파일도 복사
                    img_name = file[:-5] + '.jpg'
                    img_path = os.path.join(root, img_name)
                    if os.path.exists(img_path):
                        shutil.copy2(img_path, os.path.join(dst_dir, img_name))
                        
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    
    return counts

# 실행
src_base = '/home/minelab/바탕화면/ANN/himeow/origin_datasets'
dst_base = '/home/minelab/바탕화면/ANN/himeow/filtered_by_breeds_datasets'

# 전체 폴더에 대해 처리
total_counts = {
    'brachycephalic': {'normal': 0, 'abnormal': 0},
    'non_brachycephalic': {'normal': 0, 'abnormal': 0}
}

# 모든 폴더에 대해 실행
counts = classify_by_breeds(src_base, dst_base)

# 결과 출력
print("\n=== 데이터 통계 ===")
print("단두종:")
print(f"  - Normal: {counts['brachycephalic']['normal']}개")
print(f"  - Abnormal: {counts['brachycephalic']['abnormal']}개")
print(f"  - 총: {sum(counts['brachycephalic'].values())}개")
print("\n비단두종:")
print(f"  - Normal: {counts['non_brachycephalic']['normal']}개")
print(f"  - Abnormal: {counts['non_brachycephalic']['abnormal']}개")
print(f"  - 총: {sum(counts['non_brachycephalic'].values())}개")

total = sum(sum(type_counts.values()) for type_counts in counts.values())
print(f"\n전체 데이터 수: {total}개")