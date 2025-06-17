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

def exchange_audio(video_file, audio_file):
    """
    주어진 비디오 파일의 오디오를 교체합니다.
    
    :param video_file: 비디오 파일 경로
    :param audio_file: 새 오디오 파일 경로
    """
    import moviepy as mp
    
    # 비디오 파일과 오디오 파일을 로드
    video = mp.VideoFileClip(video_file)
    audio = mp.AudioFileClip(audio_file)
 
    # 비디오에 새 오디오를 설정
    video = video.with_audio(audio)
    
    # 새 비디오 파일로 저장
    new_video_file = video_file.replace('.mp4', '_audio_denoised.mp4')
    video.write_videofile(new_video_file, codec='libx264', audio_codec='aac')
    
    print(f"Audio replaced and saved as: {new_video_file}")

    return

# v_path = '/Volumes/One_Touch/MRI/희곡의_문학성과_공연성.mp4'
# s_path = '/Volumes/One_Touch/MRI/희곡의_문학성과_공연성_Audio_Denoise.wav'

# exchange_audio(v_path, s_path)

# 사용 예시
# directory_path = '/Volumes/One_Touch/MRI/Data/F2/out/seg'
# delete_avi_files(directory_path)