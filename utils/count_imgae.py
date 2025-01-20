import os

def count_images_in_folder(folder_path, extensions=(".jpg", ".jpeg", ".png", ".bmp", ".gif")):
    """
    폴더 내 이미지 파일 개수를 세는 함수.

    Args:
        folder_path (str): 이미지 파일을 찾을 폴더 경로.
        extensions (tuple): 이미지 파일 확장자들 (기본값: .jpg, .jpeg, .png, .bmp, .gif).

    Returns:
        int: 이미지 파일 개수.
    """
    image_count = 0
    
    # 폴더 내 파일 및 하위 폴더 순회
    for root, _, files in os.walk(folder_path):
        for file in files:
            # 파일 확장자가 이미지 확장자에 포함되는지 확인
            if file.lower().endswith(extensions):
                image_count += 1
    
    return image_count

# 예제 사용법
folder_path = "/home/minelab/desktop/ANN/Taehwa/himeow-eye/agumented_dataset/basic_brachy"
image_count = count_images_in_folder(folder_path)

print(f"폴더 내 이미지 파일 개수: {image_count}")
