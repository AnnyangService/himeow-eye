import os
import json
import shutil
from collections import defaultdict

def classify_by_breeds_and_diseases(src_directory, dst_base, disease_mapping):
    # 단두종 품종 리스트
    brachycephalic_breeds = [
        "페르시안", "히말라얀", "엑조틱 숏헤어", 
        "스코티시 폴드", "브리티시 숏헤어", "셀커크 렉스"
    ]
    
    # 목적지 디렉토리 생성 (brachy/non_brachy -> 질병 -> 유/무)
    for face_type in ["brachy", "non_brachy"]:
        for disease_kor, disease_eng in disease_mapping.items():
            for condition in ["normal", "abnormal"]:
                os.makedirs(os.path.join(dst_base, face_type, disease_eng, condition), exist_ok=True)
    
    # 카운터 초기화
    counts = defaultdict(lambda: defaultdict(lambda: {'normal': 0, 'abnormal': 0}))
    
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

                    # 유/무(normal/abnormal) 구분 (경로에서 추출)
                    if "유" in root:
                        disease_status = "abnormal"  # "유" -> 비정상
                    elif "무" in root:
                        disease_status = "normal"  # "무" -> 정상
                    else:
                        print(f"Warning: Could not determine status from path for file {file_path}")
                        continue
                    
                    # 질병 정보 추출 (경로에서 추출)
                    disease_found_kor = None
                    for disease in disease_mapping.keys():
                        if disease in root:
                            disease_found_kor = disease
                            break

                    if not disease_found_kor:
                        print(f"Warning: No disease matched in path for file {file_path}")
                        continue
                    
                    # 한글 질병 이름을 영어 이름으로 변환
                    disease_found_eng = disease_mapping.get(disease_found_kor)
                    if not disease_found_eng:
                        print(f"Warning: Disease '{disease_found_kor}' not in defined diseases for file {file_path}")
                        continue
                    
                    # 목적지 폴더 결정
                    if breed in brachycephalic_breeds:
                        dst_dir = os.path.join(dst_base, "brachy", disease_found_eng, disease_status)
                        counts["brachy"][disease_found_eng][disease_status] += 1
                    else:
                        dst_dir = os.path.join(dst_base, "non_brachy", disease_found_eng, disease_status)
                        counts["non_brachy"][disease_found_eng][disease_status] += 1
                    
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
src_base = '/home/minelab/desktop/ANN/zoo0o/himeow-eye/data/datasets/일반'
dst_base = '/home/minelab/desktop/ANN/zoo0o/himeow-eye/data/datasets/filtered_all_disease_datasets'
disease_mapping = {
    "각막부골편": "corneal_sequestrum",
    "결막염": "conjunctivitis",
    "비궤양성각막염": "non_ulcerative_keratitis",
    "안검염": "blepharitis",
    "각막궤양": "corneal_ulcer"
}

# 전체 폴더에 대해 처리
counts = classify_by_breeds_and_diseases(src_base, dst_base, disease_mapping)

# 결과 출력
print("\n=== 데이터 통계 ===")
for face_type, face_counts in counts.items():
    print(f"\n얼굴 타입: {face_type}")
    for disease, disease_counts in face_counts.items():
        print(f"  질병: {disease}")
        print(f"    - normal (무): {disease_counts['normal']}개")
        print(f"    - abnormal (유): {disease_counts['abnormal']}개")
        print(f"    - 총: {sum(disease_counts.values())}개")

total = sum(
    sum(type_counts[condition] for condition in type_counts)
    for face_counts in counts.values()
    for type_counts in face_counts.values()
)
print(f"\n전체 데이터 수: {total}개")
