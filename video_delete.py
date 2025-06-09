import os

def delete_avi_files(directory):
    """
    주어진 디렉토리 내의 모든 .avi 파일을 영구적으로 삭제합니다.
    
    :param directory: .avi 파일이 있는 디렉토리 경로
    """
    # 디렉토리의 모든 파일을 순회
    for filename in os.listdir(directory):
        # .avi 확장자가 있는 파일만 선택
        if filename.endswith('.avi'):
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

# 사용 예시
directory_path = '/Volumes/One_Touch/MRI/Data/F2/out/seg'
delete_avi_files(directory_path)