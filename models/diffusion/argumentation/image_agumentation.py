import cv2
import numpy as np
import os

def augment_image(image, output_dir, file_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 원본 이미지 저장
    cv2.imwrite(os.path.join(output_dir, f"{file_name}_original.jpg"), image)

    # 회전 (90, 180, 270도)
    for angle in [90, 180, 270]:
        rotated = rotate_image(image, angle)
        cv2.imwrite(os.path.join(output_dir, f"{file_name}_rotated_{angle}.jpg"), rotated)

    # 이동 (x축, y축으로 50픽셀씩)
    translated = translate_image(image, 50, 50)
    cv2.imwrite(os.path.join(output_dir, f"{file_name}_translated.jpg"), translated)

    # 스케일 변경 (0.5배, 1.5배)
    for scale in [0.5, 1.5]:
        scaled = scale_image(image, scale)
        cv2.imwrite(os.path.join(output_dir, f"{file_name}_scaled_{scale}.jpg"), scaled)

    # 수평 및 수직 뒤집기
    flipped_h = flip_image(image, 1)
    flipped_v = flip_image(image, 0)
    cv2.imwrite(os.path.join(output_dir, f"{file_name}_flipped_h.jpg"), flipped_h)
    cv2.imwrite(os.path.join(output_dir, f"{file_name}_flipped_v.jpg"), flipped_v)

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def translate_image(image, x, y):
    matrix = np.float32([[1, 0, x], [0, 1, y]])
    (h, w) = image.shape[:2]
    return cv2.warpAffine(image, matrix, (w, h))

def scale_image(image, scale):
    (h, w) = image.shape[:2]
    return cv2.resize(image, (int(w * scale), int(h * scale)))

def flip_image(image, flip_code):
    return cv2.flip(image, flip_code)

# 이미지 증강 실행
input_image_folder = '/home/minelab/desktop/ANN/Taehwa/himeow-eye/filtered_by_breeds_datasets/brachy/abnormal'  # 입력 이미지 폴더 경로
output_directory = '/home/minelab/desktop/ANN/Taehwa/himeow-eye/agumented_dataset/basic_brachy'  # 출력 디렉토리

# 이미지 파일 확장자 필터
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# 입력 폴더 내 모든 이미지 처리
for file_name in os.listdir(input_image_folder):
    if file_name.lower().endswith(valid_extensions):  # 이미지 파일만 필터링
        input_image_path = os.path.join(input_image_folder, file_name)
        image = cv2.imread(input_image_path)
        if image is not None:  # 이미지 파일 확인
            file_base_name = os.path.splitext(file_name)[0]
            augment_image(image, output_directory, file_base_name)