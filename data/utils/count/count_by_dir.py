import os

def list_subdirectories_and_file_count(directory):
    """
    주어진 디렉토리의 모든 하위 디렉토리와 각 디렉토리의 파일 총 개수를 출력합니다.
    """
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            # 하위 디렉토리 경로
            dir_path = os.path.join(root, dir_name)
            
            # 해당 디렉토리 내부의 파일 개수 계산
            file_count = sum(len(files) for _, _, files in os.walk(dir_path))
            
            # 디렉토리와 파일 개수 출력
            print(f"디렉토리: {dir_path}, 파일 개수: {file_count}")

# 사용 예시
if __name__ == "__main__":
    directory = input("디렉토리 경로를 입력하세요: ")
    if os.path.isdir(directory):
        print(f"{directory}의 하위 디렉토리와 파일 개수:")
        list_subdirectories_and_file_count(directory)
    else:
        print("유효하지 않은 디렉토리 경로입니다.")
