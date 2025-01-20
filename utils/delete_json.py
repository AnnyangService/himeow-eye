#데이터셋 안에 json 파일을 삭제

#목적 : 이후 코드작성시 복잡성을 줄이기 위해 전처리 단계에서 수행하기 위해 작성성

import os

def delete_json_files(directory):
    try:
        # 디렉토리 내의 파일 목록 가져오기
        for filename in os.listdir(directory):
            # 파일 경로 생성
            file_path = os.path.join(directory, filename)
            
            # .json 확장자를 가진 파일인지 확인
            if filename.endswith('.json') and os.path.isfile(file_path):
                os.remove(file_path)  # 파일 삭제
                print(f"삭제됨: {file_path}")
        
        print("JSON 파일 삭제 완료.")
    except Exception as e:
        print(f"오류 발생: {e}")

# 사용 예시
directory_path = "Taehwa/himeow-eye/filtered_by_breeds_datasets/brachy/abnormal"  # 대상 폴더 경로를 입력하세요
delete_json_files(directory_path)
