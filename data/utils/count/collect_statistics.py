import os
import json
from collections import defaultdict

def collect_statistics(directory):
    # 통계 저장
    breed_count = defaultdict(int)
    disease_count = defaultdict(int)
    
    # 디렉토리 순회
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        
                        # breed 통계 추가
                        breed = data["images"]["meta"].get("breed", "Unknown")
                        breed_count[breed] += 1
                        
                        # disease 통계 추가
                        disease = data["label"].get("label_disease_nm", "Unknown")
                        disease_count[disease] += 1
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Error reading {filepath}: {e}")
    
    return breed_count, disease_count

def print_statistics(breed_count, disease_count):
    print("\nBreed Statistics:")
    for breed, count in breed_count.items():
        print(f"{breed}: {count}")
    
    print("\nDisease Statistics:")
    for disease, count in disease_count.items():
        print(f"{disease}: {count}")

if __name__ == "__main__":
    dir_path = input("디렉토리 경로를 입력하세요: ").strip()
    if os.path.exists(dir_path):
        breed_stats, disease_stats = collect_statistics(dir_path)
        print_statistics(breed_stats, disease_stats)
    else:
        print(f"Invalid directory path: {dir_path}")
