import os
import json

def compare_folders(folder1, folder2):
    def get_files_and_metadata(folder):
        """
        Traverse the folder and extract metadata from JSON files.
        """
        data_summary = {}
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        # Extract breed and disease metadata
                        breed = data.get('images', {}).get('meta', {}).get('breed', 'Unknown')
                        disease = data.get('label', {}).get('label_disease_nm', 'Unknown')

                        # Normal/Abnormal determination based on file path
                        condition = 'normal' if 'normal' in root.lower() else 'abnormal'

                        # Add to summary
                        data_summary[file] = {
                            'path': file_path,
                            'breed': breed,
                            'disease': disease,
                            'condition': condition
                        }
                    except Exception as e:
                        print(f"Error reading JSON file {file_path}: {e}")
        return data_summary

    # Get metadata from both folders
    folder1_data = get_files_and_metadata(folder1)
    folder2_data = get_files_and_metadata(folder2)

    # Compare files
    folder1_files = set(folder1_data.keys())
    folder2_files = set(folder2_data.keys())

    only_in_folder1 = folder1_files - folder2_files
    only_in_folder2 = folder2_files - folder1_files
    common_files = folder1_files & folder2_files

    print("\n=== 비교 결과 ===")
    print(f"폴더1에만 있는 파일: {len(only_in_folder1)}개")
    for file in only_in_folder1:
        print(f"  - {file}")

    print(f"\n폴더2에만 있는 파일: {len(only_in_folder2)}개")
    for file in only_in_folder2:
        print(f"  - {file}")

    print(f"\n공통 파일: {len(common_files)}개")
    differences = []
    for file in common_files:
        meta1 = folder1_data[file]
        meta2 = folder2_data[file]
        if meta1 != meta2:
            differences.append((file, meta1, meta2))

    print(f"\n공통 파일 중 메타데이터가 다른 파일: {len(differences)}개")
    for file, meta1, meta2 in differences:
        print(f"  - 파일: {file}")
        print(f"    폴더1: {meta1}")
        print(f"    폴더2: {meta2}")

# 실행
folder1 = '/home/minelab/desktop/ANN/zoo0o/himeow-eye/data/datasets/filtered_by_breeds_datasets/brachy/normal'  # 첫 번째 폴더 경로
folder2 = '/home/minelab/desktop/ANN/zoo0o/himeow-eye/data/datasets/filtered_all_disease_datasets/brachy/non_ulcerative_keratitis/normal'  # 두 번째 폴더 경로
compare_folders(folder1, folder2)
