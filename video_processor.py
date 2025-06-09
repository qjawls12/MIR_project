import numpy as np
import subprocess
import os
import json


def mp4_to_avi(input_file, output_file):
    """
    Convert an MP4 file to AVI format without audio using FFmpeg.
    
    :param input_file: Path to the input MP4 file.
    :param output_file: Path to the output AVI file.
    """

    command = [
        'ffmpeg',
        '-i', input_file,
        '-c:v', 'libx264',  # Video codec
        output_file
    ]
    
    subprocess.run(command, check=True)

    return


def mp4_to_wav(input_file, output_file):
    """
    Convert an MP4 file to MP3 format using FFmpeg.
    
    :param input_file: Path to the input MP4 file.
    :param output_file: Path to the output MP3 file.
    """
    import subprocess

    command = [
        'ffmpeg',
        '-i', input_file,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # Audio codec
        '-ar', '16000',  # Audio sample rate
        '-ac', '1',  # Number of audio channels
        output_file
    ]
    
    subprocess.run(command, check=True)

    return



# Detect silence in a video file using FFmpeg.
def find_silence(input_file, noise=-30, duration=1):
    command = [
        'ffmpeg',
        '-i', input_file,
        '-af', f'silencedetect=noise={noise}dB:d={duration}',
        '-f', 'null',
        '-'
    ]
    
    process = subprocess.Popen(command, stderr=subprocess.PIPE, text=True)
    
    silence_events = []
    for line in process.stderr:
        if "silence_start" in line:
            start_time = float(line.split('silence_start: ')[1])
            silence_events.append({'start': start_time})
        elif "silence_end" in line:
            parts = line.split('|')
            end_time = float(parts[0].split('silence_end: ')[1])
            silence_duration = float(parts[1].split('silence_duration: ')[1])
            silence_events[-1].update({'end': end_time, 'duration': silence_duration})
    
    process.wait()
    return silence_events

# Split a video file into segments based on silence events.
def split_video(input_file, output_dir, buffer=0.1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_name = os.path.basename(input_file)
    file_name = os.path.splitext(file_name)[0]  # Remove file extension
    print(f"Splitting video: {file_name}")
    
    silence_events = find_silence(input_file)
    
    # Initialize the start of the first non-silent segment
    previous_end = 0.0
    segment_index = 1

    for event in silence_events:
        start = max(event['start'] - buffer, 0)  # Add buffer before silence
        end = event['end'] + buffer  # Add buffer after silence

        # Create segments for non-silent parts
        if previous_end < start and start > 1:
            output_file = os.path.join(output_dir, file_name + f'_segment_{segment_index:03d}.avi')
            print(f"Creating segment: {output_file} from {previous_end} to {start}")
            command = [
                'ffmpeg',
                '-i', input_file,
                '-ss', str(previous_end),
                '-to', str(start),
                '-c:v', 'libx264',  # Re-encode video
                '-c:a', 'aac',      # Re-encode audio
                output_file
            ]
            if start - previous_end > 15:
                subprocess.run(command, check=True)
                segment_index += 1
                previous_end = end

    # If there's a final segment after the last silence
    if previous_end < get_video_duration(input_file):
        output_file = os.path.join(output_dir, file_name + f'_segment_{segment_index:03d}.avi')
        print(f"Creating final segment: {output_file} from {previous_end} to end")
        command = [
            'ffmpeg',
            '-i', input_file,
            '-ss', str(previous_end),
            '-c:v', 'libx264',  # Re-encode video
            '-c:a', 'aac',      # Re-encode audio
            output_file
        ]
        subprocess.run(command, check=True)

    return

def get_video_duration(input_file):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', input_file],
        text=True, capture_output=True, check=True
    )
    return float(result.stdout.strip())



def avi_pixel_set(input_file, output_file, pixel_size = 68):
    """
    Set the pixel size of an AVI file using FFmpeg.
    
    :param input_file: Path to the input AVI file.
    :param pixel_size: Desired pixel size (width and height).
    """
    
    command = [
        'ffmpeg',
        '-i', input_file,
        '-vf', f'scale={pixel_size}:{pixel_size}',
        output_file
    ]
    
    subprocess.run(command, check=True)
    
    return


def avi_fps_set(input_file, output_file, fps=23.180):
    """
    Set the FPS of an AVI file using FFmpeg.
    
    :param input_file: Path to the input AVI file.
    :param fps: Desired frames per second.
    """

    command = [
        'ffmpeg',
        '-i', input_file,
        '-filter:v', f'fps={fps}',
        '-c:v', 'rawvideo',  # Set video codec to rawvideo
        '-vf', 'format=pal8',  # Pixel format, adjust if necessary
        '-an',  # No audio
        '-y',  # Overwrite output file if it exists
        '-r', str(fps),  # Set the output FPS
        output_file
    ]
    
    subprocess.run(command, check=True)
    
    return

def get_video_dimensions(input_file):
    """
    Get the width and height of the video using ffprobe.
    
    :param input_file: Path to the input video file.
    :return: Tuple of (width, height).
    """
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'stream=width,height', 
         '-of', 'json', input_file],
        text=True, capture_output=True, check=True
    )
    info = json.loads(result.stdout)
    width = info['streams'][0]['width']
    height = info['streams'][0]['height']
    return width, height


def avi_boundary_set(input_file, output_file):
    """
    Set the boundary of an AVI file using FFmpeg.
    
    :param input_file: Path to the input AVI file.
    :param x_min: Minimum x-coordinate.
    :param x_max: Maximum x-coordinate.
    :param y_min: Minimum y-coordinate.
    :param y_max: Maximum y-coordinate.
    """
    width, height = get_video_dimensions(input_file)
    # Define the boundary coordinates
    x_min = int(0.25 * width)  # 26% of width
    x_max = int (0.6 * width)  # 60% of width
    y_min = int(0.3 * height)  # 30% of height
    y_max = int(0.9*height)

    
    command1 = [
        'ffmpeg',
        '-i', input_file,
        '-vf', f'crop={x_max - x_min}:{y_max - y_min}:{x_min}:{y_min}',
        '-an',
        output_file
    ]
    
    subprocess.run(command1, check=True)

    # brightness and contrast adjustment
    command2 = [
        'ffmpeg',
        '-i', output_file,
        '-vf', 'eq=brightness=0.05:contrast=1.2',
        '-c:v', 'libx264',  # Video codec
        '-an',  # No audio
        output_file
    ]
    subprocess.run(command2, check=True)
    
    return

if __name__ == "__main__":
    # Example usage
    base_dir = os.path.dirname('/Volumes/One_Touch/rtMRI_Dataset/VOCOLAB/')

    mp4_dir = os.path.join(base_dir,'mp4')
    mp4_to_avi_dir = os.path.join(base_dir,'mp4_to_avi')
    wav_dir = os.path.join(base_dir,'wav')
    avi_split_dir = os.path.join(base_dir,'avi_split')
    avi_crop_dir = os.path.join(base_dir,'avi_crop')
    avi_pixel_dir = os.path.join(base_dir,'avi_pixel')
    avi_fps_dir = os.path.join(base_dir,'avi')
    print(f"Base directory: {base_dir}")
    print(f"MP4 directory: {mp4_dir}")
    print(f"AVI directory: {mp4_to_avi_dir}")
    print(f"AVI Split directory: {avi_split_dir}")
    print(f"WAV directory: {wav_dir}")
    print(f"AVI Crop directory: {avi_crop_dir}")
    print(f"AVI Pixel directory: {avi_pixel_dir}")
    print(f"AVI FPS directory: {avi_fps_dir}")

    # Define pixel size and FPS
    pixel_size = 68
    fps = 23.180


    if not os.path.exists(mp4_dir):
        os.makedirs(mp4_dir)
    if not os.path.exists(mp4_to_avi_dir):
        os.makedirs(mp4_to_avi_dir)
    if not os.path.exists(avi_split_dir):
        os.makedirs(avi_split_dir)
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)
    if not os.path.exists(avi_crop_dir):
        os.makedirs(avi_crop_dir)
    if not os.path.exists(avi_pixel_dir):
        os.makedirs(avi_pixel_dir)
    if not os.path.exists(avi_fps_dir):
        os.makedirs(avi_fps_dir)

    # # Split MP4 files into AVI and WAV with segments based on silence
    # for file in os.listdir(mp4_dir):
    #     if file.endswith('.mp4') and not file.startswith('._'):
    #         input_mp4 = os.path.join(mp4_dir, file)
    #         output_avi = os.path.join(mp4_to_avi_dir, file.replace('.mp4','.avi'))
            
    #         # Process each MP4 file
    #         print(f"Processing {input_mp4}...")
            
    #         # Convert MP4 to AVI
    #         if not os.path.exists(output_avi):
    #             print(f"Converting {input_mp4} to {output_avi}...")
    #             mp4_to_avi(input_mp4, output_avi)

    # for file in os.listdir(mp4_to_avi_dir):
    #     if file.endswith('.avi') and not file.startswith('._'):
    #         input_avi = os.path.join(mp4_to_avi_dir, file)

    #         print(f"Splitting {input_avi}...")
    #         split_video(input_avi, avi_split_dir, buffer=0.1)

    

    #Process all split AVI files in the directory
    for file in os.listdir(avi_split_dir):
        if file.endswith('.avi') and not file.startswith('._'):
            input_avi = os.path.join(avi_split_dir, file)
            output_wav = os.path.join(base_dir, file.split('_')[0], 'wav', file.replace('.avi', '.wav'))
            if not os.path.exists(os.path.join(base_dir, file.split('_')[0], 'wav')):
                os.makedirs(os.path.join(base_dir, file.split('_')[0], 'wav'))
            output_avi_crop = os.path.join(avi_crop_dir, file.replace('.avi', '_crop.avi'))
            output_pixel_avi = os.path.join(avi_pixel_dir, file.replace('.avi', f'_pixel_{pixel_size}.avi'))
            output_fps_avi = os.path.join(avi_fps_dir, file.split('_')[0] , file.replace('.mp4', f'_{fps}fps.avi'))
            if not os.path.exists(os.path.join(avi_fps_dir, file.split('_')[0])):
                os.makedirs(os.path.join(avi_fps_dir, file.split('_')[0]))
            
            # Process each Splitted AVI file
            print(f"Processing {input_avi}...")

            # Convert AVI to WAV
            if not os.path.exists(output_wav):
                print(f"Converting {input_avi} to {output_wav}...")
                mp4_to_wav(input_avi, output_wav)

            # Set boundary of AVI
            if not os.path.exists(output_avi_crop):
                print(f"Setting boundary for {input_avi}...")
                avi_boundary_set(input_avi, output_avi_crop)
   
            # Set pixel size of AVI
            if not os.path.exists(output_pixel_avi):
                print(f"Setting pixel size for {input_avi} to {pixel_size} x {pixel_size}...")
                avi_pixel_set(output_avi_crop, output_pixel_avi, pixel_size=pixel_size)

            # Set FPS of AVI
            if not os.path.exists(output_fps_avi):
                print(f"Setting FPS for {output_pixel_avi} to {fps}...")
                avi_fps_set(output_pixel_avi, output_fps_avi, fps=fps)

            print(f"Finished processing {file}.\n")
    print("All files processed successfully.")