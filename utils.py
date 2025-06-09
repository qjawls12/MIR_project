import os
import torch
import torchaudio
import librosa
import numpy as np
import scipy.io
from scipy.io import wavfile
import matplotlib.pyplot as plt
import IPython.display as IPD
import pandas as pd

# 파형 시각화 함수
def plot_waveform(waveform, sample_rate):
    plt.figure(figsize=(12, 4))
    plt.plot(waveform.t().numpy())
    plt.title('Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.xlim(0, waveform.shape[1])
    plt.show()

# Load audio and .mat file
def load_audio_and_vt_data(wav_path, sample_rate=16000):
    """
    Load audio and .mat file from given paths.
    
    Parameters:
    wav_path (str): Path to the .wav file.
    mat_path (str): Path to the .mat file.
    
    Returns:
    tuple: (audio_tensor, sample_rate, mat_data)
    """
    mat_path = wav_path.replace('.wav', '_morph.mat')  # Assuming .mat file has the same name as .wav file
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio file not found at {wav_path}")
    
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f".mat file not found at {mat_path}")
    
    audio_tensor, sr = torchaudio.load(wav_path)
    if sr != sample_rate:
        audio_tensor = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(audio_tensor)
    audio_tensor = audio_tensor.mean(dim=0, keepdim=True)  # Convert to mono if stereo
    mat = scipy.io.loadmat(mat_path)
    vt_data1 = torch.tensor(mat['pts_lb_c'], dtype=torch.float32)
    vt_data2 = torch.tensor(mat['pts_rt_c'], dtype=torch.float32)
    vt_data1[vt_data1 > 100] = 0  # Filter out values greater than 100
    vt_data2[vt_data2 > 100] = 0  # Filter out values greater than 100
    mat_data = {
        'inner_tract': vt_data1,
        'outer_tract': vt_data2
    }
    return audio_tensor, mat_data

# Estimate fundamental frequency (F0) using librosa's pyin function
def estimate_fundamental_frequency(audio_tensor, sample_rate=16000):
    """
    Estimate the fundamental frequency (F0) from the audio tensor.
    
    Parameters:
    audio_tensor (torch.Tensor): Audio tensor.
    sample_rate (int): Sample rate of the audio.
    
    Returns:
    np.ndarray: Estimated F0 values.
    """
    audio_np = audio_tensor.numpy()
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=0)  # Convert to mono if stereo
    audio_np = np.concatenate(np.zeros(345), audio_np, np.zeros(1600),axis=0)  # Add padding to match librosa's expected input shape
    f0, voiced_flag, voiced_probs = librosa.pyin(audio_np, fmin=50, fmax=1000,
                                                 sr=sample_rate, frame_length = 1382, hop_length = 691)
    return torch.tensor(f0, dtype=torch.float32)

# Estimate Mel spectrogram from audio tensor
def get_mel_spectrogram(audio_tensor, sample_rate=16000, n_mels=128):
    """
    Compute the Mel spectrogram of the audio tensor.
    
    Parameters:
    audio_tensor (torch.Tensor): Audio tensor.
    sample_rate (int): Sample rate of the audio.
    n_mels (int): Number of Mel bands.
    
    Returns:
    torch.Tensor: Mel spectrogram.
    """
    audio_tensor = torch.cat((torch.zeros(345), audio_tensor, torch.zeros(1600)), dim=0)  # Add padding to match librosa's expected input shape
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, win_length=1382, hop_length = 691, n_mels=n_mels)
    return mel_transform(audio_tensor)

# Estimate sound pressure level (SPL) from Mel spectrogram
def estimate_spectrogram_spl(audio, sample_rate=16000):
    """
    Estimate the sound pressure level (SPL) from the Mel spectrogram.
    
    Parameters:
    mel_spectrogram (torch.Tensor): Mel spectrogram tensor.
    sample_rate (int): Sample rate of the audio.
    
    Returns:
    torch.Tensor: Estimated SPL values.
    """
    mel_spectrogram = get_mel_spectrogram(audio, sample_rate=sample_rate)
    # Convert Mel spectrogram to dB scale
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
    # Calculate SPL
    log_mel = 20 * torch.log10(mel_db + 1e-6)  # Adding a small value to avoid log(0)
    spl = log_mel.mean(dim=-1)  # Average across frequency bins
    return spl

# label data
def label_vt_f0_spl(wav_path, csv_dir, sample_rate=16000):
    """
    Label the vocal tract data with fundamental frequency (F0) and sound pressure level (SPL).
    
    Parameters:
    vt_data (torch.Tensor): Vocal tract data tensor.
    f0_data (torch.Tensor): Fundamental frequency data tensor.
    spl_data (torch.Tensor): Sound pressure level data tensor.
    
    Returns:
    dict: Dictionary containing labeled data.
    """
    file_name = os.path.basename(wav_path).replace('.wav', '.csv')
    audio_tensor, mat_data = load_audio_and_vt_data(wav_path = wav_path, sample_rate=sample_rate)
    inner_tract = mat_data['inner_tract']
    outer_tract = mat_data['outer_tract']
    f0 = estimate_fundamental_frequency(audio_tensor, sample_rate)
    spl = estimate_spectrogram_spl(audio_tensor, sample_rate)
    if inner_tract.shape[0] < f0.shape[0]:
        f0 = f0[:inner_tract.shape[0]]  # Truncate F0 to match inner_tract length
    elif inner_tract.shape[0] > f0.shape[0]:
        f0 = torch.cat((f0, torch.zeros(inner_tract.shape[0] - f0.shape[0])), dim=0)
    if inner_tract.shape[0] < spl.shape[0]:
        spl = spl[:inner_tract.shape[0]]  # Truncate SPL to match inner_tract length
    elif inner_tract.shape[0] > spl.shape[0]:
        spl = torch.cat((spl, torch.zeros(inner_tract.shape[0] - spl.shape[0])), dim=0)
    
    
    csv_dict = {
        'wav_path': wav_path,
        'inner_tract': inner_tract,
        'outer_tract': outer_tract,
        'f0': f0,
        'spl': spl
    }
    pd.DataFrame(csv_dict).to_csv(os.path.join(csv_dir, file_name), index=False)

    return


# Calculate vocal tract length
def calculate_vt_dl(est_tensor):
    """
    Calculate the length of the vocal tract data.
    
    Parameters:
    vt_data (torch.Tensor): Vocal tract data tensor.
    
    Returns:
    float: Length of the vocal tract.
    """
    assert est_tensor.ndim == 3, "Input tensor must be 3-dimensional (batch, time, features)"
    tract_tensor = est_tensor[:,:,6:]  # Clone the input tensor to avoid modifying the original
    tract_tensor = tract_tensor.reshape(tract_tensor.shape[0],tract_tensor.shape[1], -1, 4)  # Reshape to (batch, time, 2) if needed
    vt_data1 = tract_tensor[:,:,:,:2] # Assuming you want to calculate length for inner_tract
    vt_data2 = tract_tensor[:,:,:,2:-1]  # Assuming you want to calculate length for outer_tract
    vt_center = (vt_data1 + vt_data2) / 2  # Average of inner and outer tracts
    vt_dpts = vt_center[:, :, 1:,:] - vt_center[:, :, :-1,:]  # Calculate differences between consecutive points
    vt_dl = torch.sqrt(torch.sum(vt_dpts ** 2, dim=-1))  # Euclidean distance for each segment
    
    return vt_dl




class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir:str, csv_dir:str , sample_rate=16000):
        super(AudioDataset, self).__init__()
        self.data_dir = data_dir
        self.csv_dir = csv_dir
        self.sample_rate = sample_rate
        if not os.path.exists(self.csv_dir):
            self.save_data_to_csv()
        self.csv_path_list = self.load_csv_path()
        if not self.csv_path_list:
            raise ValueError("No CSV files found in the specified directory.")

    def __len__(self):   
        return len(self.csv_path_list)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of range for the dataset.")
        csv_path = self.csv_path_list[idx]
        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError(f"CSV file at {csv_path} is empty.")
        row = df.iloc[0]  # Assuming we want the first row of the CSV file
        wav_file = row['wav_path']
        if not os.path.exists(wav_file):
            raise FileNotFoundError(f"WAV file not found at {wav_file}")
        
        # Load audio file
        audio = torchaudio.load(wav_file)  # Load audio file
        if audio.shape[0] > 1:
            audio_tensor = audio.mean(dim=0, keepdim=True)  # Convert to mono if stereo
        
        # load labels
        inner_tract = row['inner_tract']  # Assuming inner_tract is the label we want
        outer_tract = row['outer_tract']
        f0 = row['f0']  # Fundamental frequency
        spl = row['spl']  # Sound pressure level
        if isinstance(inner_tract, str):
            inner_tract = np.fromstring(inner_tract[1:-1], sep=',')
        if isinstance(outer_tract, str):
            outer_tract = np.fromstring(outer_tract[1:-1], sep=',')
        if isinstance(f0, str):
            f0 = np.fromstring(f0[1:-1], sep=',')
        if isinstance(spl, str):
            spl = np.fromstring(spl[1:-1], sep=',')
        inner_tract = torch.tensor(inner_tract, dtype=torch.float32)
        outer_tract = torch.tensor(outer_tract, dtype=torch.float32)
        f0 = torch.tensor(f0, dtype=torch.float32)
        spl = torch.tensor(spl, dtype=torch.float32)

        assert audio_tensor.shape[0] == 1, "Audio tensor should be mono."
        assert audio_tensor.shape[1] > 0, "Audio tensor should not be empty."
        assert inner_tract.ndim == 3, "Inner tract data should be 2-dimensional."
        assert outer_tract.ndim == 3, "Outer tract data should be 2-dimensional."
        assert f0.ndim == 1, "F0 data should be 1-dimensional."
        assert spl.ndim == 1, "SPL data should be 1-dimensional."
        assert inner_tract.shape[0] == outer_tract.shape[0], "Inner and outer tract data should have the same number of frames."
        assert inner_tract.shape[0] == f0.shape[0], "Inner tract and F0 data should have the same number of frames."
        assert inner_tract.shape[0] == spl.shape[0], "Inner tract and SPL data should have the same number of frames."

        return audio_tensor, (inner_tract, outer_tract, f0, spl)  # Return audio tensor and labels as a tuple
    
    def load_csv_path(self):
        if not os.path.exists(self.csv_dir):
            raise FileNotFoundError(f"CSV file not found at {self.csv_dir}")
        csv_path_list = [os.path.join(self.csv_dir, file) for file in os.listdir(self.csv_dir) if file.endswith('.csv')]
        return csv_path_list
    
    def save_data_to_csv(self):
        """        Save audio and vocal tract data to CSV files.
        """
        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)
        for file in os.listdir(self.data_dir):
            if file.endswith('.wav'):
                wav_path = os.path.join(self.wav_dir, file)
                # Load audio and vocal tract data
                label_vt_f0_spl(wav_path, self.csv_dir, sample_rate=self.sample_rate)
        print(f"Data saved to CSV files in {self.csv_dir}")
        return
    






# 데이터셋 열람
path = '/Volumes/One_Touch/rtMRI_Dataset/VOCOLAB/Singer5/out/morph'
mat = scipy.io.loadmat(os.path.join(path,'Singer5_segment_001_morph.mat'))
# mat = scipy.io.loadmat('/Users/parkbeomjin/Desktop/막학기/Deep learning for Music & Audio/프로젝트/MIR_project/span_segmentation/demo_files/template_struct.mat')

wav_path = '/Volumes/One_Touch/rtMRI_Dataset/VOCOLAB/Singer5/out/wav'
print(os.path.exists(os.path.join(wav_path,'Singer5_segment_001.wav')))
y, sr = torchaudio.load(os.path.join(wav_path,'Singer5_segment_001.wav'))
print(y.shape, sr)  # 오디오 데이터의 형태와 샘플링 레이트 출력
IPD.display(IPD.Audio(os.path.join(wav_path,'Singer5_segment_001.wav')))  # 오디오 재생


    
print('Dataset loaded successfully!')

plot_waveform(y, sr)  # 오디오 파형 시각화
print('Keys in the dataset:', mat.keys())  # 데이터셋의 키 출력
print(mat['pts_lb_c'])
print(mat['pts_lb_c'].shape)  # 데이터의 형태 출력
print(mat['pts_rt_c'].shape)  # 첫 번째 항목의 데이터 추출
print(mat['vtl'].shape)  # 첫 번째 항목의 데이터 추출
print(type(mat['pts_rt_c'][0,0,0]))  # 첫 번째 항목의 데이터 추출
row_indices, col_indices = np.where(mat['pts_lb_c'][0] < 100) # 첫 번째 열이 1인 행의 인덱스
print([(row, col) for row, col in zip(row_indices, col_indices)]) # 행과 열의 인덱스 출력
print(mat['pts_lb_c'][0][mat['pts_lb_c'][0] < 100].shape)  # 첫 번째 열이 1인 행의 인덱스에 해당하는 값 출력
arr = mat['pts_lb_c'][52]
arr2 = mat['pts_rt_c'][52]
arr3 = mat['vtl'][52]
result = np.apply_along_axis(lambda x: x[x < 100], 0, arr)
result2 = np.apply_along_axis(lambda x: x[x < 100], 0, arr2) # 각 열에서 100 미만의 값만 추출
result3 = np.apply_along_axis(lambda x: x[x < 100], 0, arr3) # 각 열에서 100 미만의 값만 추출
print(result.shape)  # 결과의 형태 출력  # 결과 출력
print(result2.shape)  # 결과의 형태 출력
print(result3.shape)  # 결과 출력
x = result[:, 0]  # 첫 번째 열의 첫 번째 행의 값
y = result[:, 1]  # 첫 번째 열의 두 번째 행의 값
plt.plot(x, y, linestyle='-', color='b')
plt.plot(result2[:, 0], result2[:, 1], linestyle='-', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Plot of pts_lb_c')
plt.gca().invert_yaxis()  # y축 반전
plt.show()

























"""
# 예시 좌표 데이터 가져오기
curves = mat['template_struct'][0,2]['template'][0,0]['curves']
for i in range(curves.shape[1]):  # curves 데이터 접근
  coords = curves[0,i]['position'] # 첫 번째 좌표 배열

# 좌표 시각화
  x = coords[:, 0]  # x 좌표
  y = coords[:, 1]  # y 좌표
  plt.plot(x, y)
plt.show() # 데이터셋의 키 출력
#print(mat['trackdata'].dtype)  # 데이터의 형태 출력
#print(type(mat['trackdata']))           # <class 'numpy.ndarray
#print(type(mat['trackdata'][0,0]))    # 구조체라면 <class 'numpy.void'> 또는 object
#first_item = mat['trackdata'][0, 0]
#print(type(first_item))
#print(first_item.shape)   # 내부 데이터 형태
#print(first_item.dtype)   # 자료형
#data = mat['trackdata'][0, 200]  # 첫 번째 항목의 데이터 추출
# print(data['contours'])
# print(data['frameNo'])
# print(first_item[:5])
#print(data)  # 첫 번째 프레임의 첫 번째 객체의 contour 좌표
#data['contours'][0, 0][0][0]['i']  # 군집 또는 클래스 인덱스
#data['contours'][0, 0][0][0]['mu']
#print(type(data['contours']))
#print(type(data['frameNo']))
#print(data['contours'][0,0][0,0]['segment'][0,0]['v'])
  # 첫 번째 프레임의 첫 번째 객체의 segment 좌표
#print(data['frameNo'][0,0][0,0]) # mu의 shape 확인
length_list = []
for i in range(mat['trackdata'].shape[1]):
    data = mat['trackdata'][0, i]
    contour = data['contours'][0,0][0,0]['segment']
    coords_concat = np.concatenate([contour[0,j]['v'][0,0] for j in range(contour.shape[1]-1)], axis=0).shape
    length_list.append(coords_concat)
# print(length_list)
# print(len(length_list)) # 각 프레임의 contour 좌표 길이 출력
data = mat['trackdata'][0, 15]
contour = data['contours'][0,0][0,0]['segment']
frame = data['frameNo'][0,0][0,0]

for i in range(contour.shape[1]-1):
    coords = contour[0,i]['v'][0,0]
    # print(coords)
    # j = 0
    # while (j < coords.shape[0]):
    #     if coords[j, 0] > 25:
    #         print(coords[j,0])
    #         coords = np.delete(coords, j, axis=0)
    #     else:
    #         j += 1
    coords = coords[coords[:, 0] <= 22]
    coords = coords[coords[:, 1] <= 30]
    x = coords[:,0]
    y = coords[:,1]
    # print(coords.shape)
    plt.plot(x,y, label='Contour', color = 'orange', linewidth=1)
# coords_concat = np.concatenate([contour[0,i]['v'][0,0] for i in range(contour.shape[1]-1)], axis=0)
# x = coords_concat[:,0]
# y = coords_concat[:,1]
# plt.plot(x, y, label='Concatenated Contour', color='green', linewidth=2)


plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'frame_No.{frame}')
plt.legend()
plt.show()
"""
