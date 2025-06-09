import subprocess
import json

def get_video_codec_info(file_path):
    # ffprobe 명령어를 실행하여 비디오 파일의 스트림 정보를 JSON 형식으로 출력
    command = [
        'ffprobe',
        '-v', 'error',  # 오류 메시지만 출력
        '-select_streams', 'v:0',  # 첫 번째 비디오 스트림 선택
        '-show_entries', 'stream=codec_name',  # 코덱 이름만 출력
        '-of', 'json',  # 출력 형식을 JSON으로 설정
        file_path
    ]
    
    # subprocess를 사용하여 명령어 실행
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # JSON 형식의 출력을 파싱
    try:
        info = json.loads(result.stdout)
        codec_name = info['streams'][0]['codec_name']
        return codec_name
    except (json.JSONDecodeError, KeyError, IndexError):
        return None

import subprocess

def convert_to_rawvideo(input_file, output_file):
    """
    Convert a video file to rawvideo format using FFmpeg.
    
    :param input_file: Path to the input video file with mpeg4 codec.
    :param output_file: Path to save the output video file with rawvideo codec.
    """
    command = [
        'ffmpeg',
        '-i', input_file,
        '-c:v', 'rawvideo',  # Set video codec to rawvideo
        '-pix_fmt', 'yuv420p',  # Pixel format, adjust if necessary
        output_file
    ]
    
    # Execute the command
    try:
        subprocess.run(command, check=True)
        print(f"Conversion successful: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")


# 사용 예시
file_path1 = '/Volumes/One Touch/rtMRI Dataset/VOCOLAB/avi/Singer 1/Singer 1_segment_001.avi'
codec = get_video_codec_info(file_path1)
if codec:
    print(f"비디오 코덱: {codec}")
else:
    print("코덱 정보를 가져오는 데 실패했습니다.")

file_path2 = '/Volumes/One Touch/MRI/Data/F1/avi/usctimit_mri_f1_001_005.avi'
codec = get_video_codec_info(file_path2)
if codec:
    print(f"비디오 코덱: {codec}")
else:
    print("코덱 정보를 가져오는 데 실패했습니다.")