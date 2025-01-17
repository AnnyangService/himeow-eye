import os
import shutil
from sklearn.model_selection import train_test_split

def create_directory_structure(base_dir):
    """train, val, test 디렉토리를 생성합니다."""
    datasets_dir = os.path.join(base_dir, "datasets", "images")  # datasets/images 하위 디렉토리 생성
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(datasets_dir, split)
        os.makedirs(os.path.join(split_dir, "abnormal"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "normal"), exist_ok=True)

def split_and_copy_files(source_dir, dest_dir, test_size=0.1, val_size=0.2):
    """데이터셋을 train, val, test로 나누고 파일을 복사합니다."""
    datasets_dir = os.path.join(dest_dir, "datasets", "images")  # datasets/images 경로 지정
    for category in ["abnormal", "normal"]:
        category_dir = os.path.join(source_dir, category)
        files = os.listdir(category_dir)

        # 먼저 train + val과 test를 나눕니다.
        train_val_files, test_files = train_test_split(files, test_size=test_size, random_state=42)

        # train과 val을 나눕니다.
        train_files, val_files = train_test_split(train_val_files, test_size=val_size / (1 - test_size), random_state=42)

        # 파일을 해당 디렉토리에 복사합니다.
        for split, split_files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            split_category_dir = os.path.join(datasets_dir, split, category)
            for file_name in split_files:
                src_file = os.path.join(category_dir, file_name)
                dest_file = os.path.join(split_category_dir, file_name)
                shutil.copy(src_file, dest_file)

if __name__ == "__main__":
    # 소스 디렉토리와 대상 디렉토리 설정
    source_directory = "/home/minelab/바탕화면/ANN/zoo0o/himeow-eye/data/datasets/filtered_by_breeds_datasets/brachy"
    destination_directory = "/home/minelab/바탕화면/ANN/zoo0o/himeow-eye/data/datasets_yolo/split_by_brachy_datasets"

    # train/val/test 디렉토리 구조 생성
    create_directory_structure(destination_directory)

    # 데이터셋 나누고 파일 복사
    split_and_copy_files(source_directory, destination_directory)
