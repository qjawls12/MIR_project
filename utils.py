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
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pad_packed_sequence, PackedSequence
import torch.nn as nn
import torch.nn.functional as F

# 파형 시각화 함수
def plot_waveform(waveform, sample_rate=16000):
    plt.figure(figsize=(12, 4))
    plt.plot(waveform.t().numpy())
    plt.title('Waveform')
    plt.xlabel('Sample')
    plt.xticks(np.arange(0, waveform.shape[1], sample_rate), np.arange(0, waveform.shape[1] / sample_rate, 1))
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
    audio_tensor = audio_tensor.mean(dim=0, keepdim=False)  # Convert to mono if stereo
    mat = scipy.io.loadmat(mat_path)
    vt_image = np.zeros((mat['pts_lb_c'].shape[0], 136, 136), dtype=np.uint8)  # 136x136 크기의 빈 이미지 생성
    for i in range(mat['pts_lb_c'].shape[0]):
        vt_data1 = np.apply_along_axis(lambda x: x[x < 100], 0, mat['pts_lb_c'][i])
        vt_data2 = np.apply_along_axis(lambda x: x[x < 100], 0, mat['pts_rt_c'][i])  # Filter out values greater than 100
        vt_data1 = 2*vt_data1.astype(np.uint8)  # Convert to int16
        vt_data2 = 2*vt_data2.astype(np.uint8)  # Convert to int16
        vt_image[i, vt_data1[:, 0], vt_data1[:, 1]] = 255  # 좌표에 해당하는 픽셀 값 설정
        vt_image[i, vt_data2[:, 0], vt_data2[:, 1]] = 255  # 좌표에 해당하는 픽셀 값 설정
    vt_image = vt_image.astype(np.uint8)  # Convert to uint8
    
    return audio_tensor.numpy(), vt_image
    
    assert audio_tensor.ndim == 1, "Audio tensor should be 1-dimensional (channels, samples)"
    return audio_tensor.numpy(), mat_data

# Estimate fundamental frequency (F0) using librosa's pyin function
def estimate_fundamental_frequency(audio_np, sample_rate=16000):
    """
    Estimate the fundamental frequency (F0) from the audio tensor.
    
    Parameters:
    audio_tensor (torch.Tensor): Audio tensor.
    sample_rate (int): Sample rate of the audio.
    
    Returns:
    np.ndarray: Estimated F0 values.
    """
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=0)  # Convert to mono if stereo
    audio_np = np.concatenate([np.zeros(345), audio_np, np.zeros(1600)],axis=0)  # Add padding to match librosa's expected input shape
    f0, voiced_flag, voiced_probs = librosa.pyin(audio_np, fmin=50, fmax=1000,
                                                 sr=sample_rate, frame_length = 1382, hop_length = 691)
    f0 = np.nan_to_num(f0, nan=0.0)  # Replace NaN values with 0
    # print(np.isnan(f0).any(), np.isnan(voiced_flag).any(), np.isnan(voiced_probs).any())
    return f0

# Estimate spectrogram from audio tensor
def get_spectrogram(audio_np, sample_rate=16000):
    """
    Compute the spectrogram of the audio tensor.
    
    Parameters:
    audio_tensor (torch.Tensor): Audio tensor.
    sample_rate (int): Sample rate of the audio.
    
    
    Returns:
    torch.Tensor: spectrogram.
    """
    audio = torch.tensor(audio_np).to(dtype=torch.float32)
    if audio.ndim > 1:
        audio = audio.mean(dim=0)  # Convert to mono if stereo
    audio_tensor = torch.cat((torch.zeros(345), audio, torch.zeros(1600)), dim=0)  # Add padding to match librosa's expected input shape
    transform = torchaudio.transforms.Spectrogram(n_fft = 1382, win_length=1382, hop_length = 691)
    spec = transform(audio_tensor)
    spec = spec.transpose(0, 1)
    return spec

# Estimate sound pressure level (SPL) from spectrogram
def estimate_spectrogram_spl(audio, sample_rate=16000):
    """
    Estimate the sound pressure level (SPL) from th spectrogram.
    
    Parameters:
    spectrogram (torch.Tensor): spectrogram tensor.
    sample_rate (int): Sample rate of the audio.
    
    Returns:
    torch.Tensor: Estimated SPL values.
    """
    spectrogram = get_spectrogram(audio, sample_rate=sample_rate)
    # Convert spectrogram to dB scale
    spec_db = torchaudio.transforms.AmplitudeToDB()(spectrogram)
    # Calculate SPL
    spl = spec_db.mean(dim=-1) # Average across frequency bins
    # print(spl.isnan().any(), spl.isinf().any())
    return spl.numpy()  # Convert to numpy array for consistency

# label data
def label_vt_f0_spl(wav_path, npz_dir, sample_rate=16000):
    """
    Label the vocal tract data with fundamental frequency (F0) and sound pressure level (SPL).
    
    Parameters:
    vt_data (torch.Tensor): Vocal tract data tensor.
    f0_data (torch.Tensor): Fundamental frequency data tensor.
    spl_data (torch.Tensor): Sound pressure level data tensor.
    
    Returns:
    dict: Dictionary containing labeled data.
    """
    file_name = os.path.basename(wav_path).replace('.wav', '.npz')
    audio_np, vt_image = load_audio_and_vt_data(wav_path = wav_path, sample_rate=sample_rate)
    f0 = estimate_fundamental_frequency(audio_np, sample_rate)
    spl = estimate_spectrogram_spl(audio_np, sample_rate)
    if vt_image.shape[0] < f0.shape[0]:
        f0 = f0[:vt_image.shape[0]]  # Truncate F0 to match inner_tract length
    elif vt_image.shape[0] > f0.shape[0]:
        f0 = np.concatenate((f0, np.zeros(vt_image.shape[0] - f0.shape[0])), axis=0)
    if vt_image.shape[0] < spl.shape[0]:
        spl = spl[:vt_image.shape[0]]  # Truncate SPL to match inner_tract length
    elif vt_image.shape[0] > spl.shape[0]:
        spl = np.concatenate((spl, np.zeros(vt_image.shape[0] - spl.shape[0])), axis=0)
    
    audio = np.concatenate((np.zeros(345), audio_np, np.zeros(1600)), axis=0)  # Add padding to match librosa's expected input shape
    
    npz_dict = {
        'audio': audio,
        'vt_image': vt_image,  # Vocal tract image
        'f0': f0,
        'spl': spl
    }
    np.savez(os.path.join(npz_dir, file_name), **npz_dict)  # Save to .npz file
    print(f"Data saved to {os.path.join(npz_dir, file_name)}")

    return

def pack_collate_fn(batch: list):
    """
    Custom collate function to handle variable-length sequences in a batch.
    
    Parameters:
    batch (list): List of tuples containing audio tensors and labels.
    
    Returns:
    tuple: Packed sequences for audio and labels.
    """
    audio_tensors, labels = zip(*batch)
    audio_packed = pack_sequence(audio_tensors, enforce_sorted=False)
    
    vt_image, f0s, spls = zip(*labels)
    vt_image_packed = pack_sequence(vt_image, enforce_sorted=False)
    f0_packed = pack_sequence(f0s, enforce_sorted=False)
    spl_packed = pack_sequence(spls, enforce_sorted=False)
    
    return audio_packed, (vt_image_packed, f0_packed, spl_packed)



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
    def __init__(self, data_dir:str, npz_dir:str , sample_rate=16000):
        super(AudioDataset, self).__init__()
        self.data_dir = data_dir
        self.npz_dir = npz_dir
        self.sample_rate = sample_rate
        if not os.path.exists(self.npz_dir) or not os.listdir(self.npz_dir):
            self.save_data_to_npz()
        self.npz_path_list = self.load_npz_path()
        if not self.npz_path_list:
            raise ValueError("No npz files found in the specified directory.")

    def __len__(self):   
        return len(self.npz_path_list)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of range for the dataset.")
        npz_path = self.npz_path_list[idx]
        df = np.load(npz_path)  # Load the .npz file
        if df is None or len(df) == 0:
            raise ValueError(f"CSV file at {npz_path} is empty.")
        audio = df['audio']  # Load audio tensor from the .npz file
        # audio_tensor = audio_tensor.unsqueeze(0) if audio_tensor.ndim == 1 else audio_tensor  # Ensure audio tensor is 2D (batch, samples)
        # load labels
        vt_image = df['vt_image']  # Assuming inner_tract is the label we want
        f0 = df['f0']  # Fundamental frequency
        spl = df['spl']  # Sound pressure level

        # assert audio_tensor.shape[0] == 1, "Audio tensor should be mono."
        # assert audio_tensor.shape[1] > 0, "Audio tensor should not be empty."
        assert vt_image.ndim == 3, "Inner tract data should be 2-dimensional."
        assert f0.ndim == 1, "F0 data should be 1-dimensional."
        assert spl.ndim == 1, "SPL data should be 1-dimensional."
        assert vt_image.shape[0] == f0.shape[0], "Inner tract and F0 data should have the same number of frames."
        assert vt_image.shape[0] == spl.shape[0], "Inner tract and SPL data should have the same number of frames."
        assert vt_image.shape[0] == f0.shape[0], "Inner tract and audio data should have the same number of frames."
        audio = torch.tensor(audio, dtype=torch.float32)  # Convert audio to tensor
        vt_image = torch.tensor(vt_image, dtype=torch.float32)
        f0 = torch.tensor(f0, dtype=torch.float32)
        spl = torch.tensor(spl, dtype=torch.float32)

        return audio, (vt_image, f0, spl)  # Return audio tensor and labels as a tuple
    
    def load_npz_path(self):
        if not os.path.exists(self.npz_dir):
            raise FileNotFoundError(f"npz file not found at {self.npz_dir}")
        npz_path_list = [os.path.join(self.npz_dir, file) for file in os.listdir(self.npz_dir) if file.endswith('.npz')]
        return npz_path_list
    
    def save_data_to_npz(self):
        """        Save audio and vocal tract data to CSV files.
        """
        if not os.path.exists(self.npz_dir):
            os.makedirs(self.npz_dir)
        for file in os.listdir(self.data_dir):
            if file.endswith('.wav'):
                wav_path = os.path.join(self.data_dir, file)
                # Load audio and vocal tract data
                label_vt_f0_spl(wav_path, self.npz_dir, sample_rate=self.sample_rate)
        print(f"Data saved to npz files in {self.npz_dir}")
        return
    


# loss functions
def vt_loss(est_tensor, target_tensor):
    """
    Calculate the loss between estimated and target vocal tract data.
    
    Parameters:
    est_tensor (torch.Tensor): Estimated vocal tract data tensor.
    target_tensor (torch.Tensor): Target vocal tract data tensor.
    
    Returns:
    torch.Tensor: Loss value.
    """
    target, lengths = pad_packed_sequence(target_tensor, batch_first=True)
    target = target.reshape(target.shape[0], target.shape[1], -1)  # Reshape (batch, time, w, h) to (batch, time, w*h)
    assert est_tensor.shape[0] == target.shape[0], "Estimated and target tensors must have the same shape."
    loss = torch.zeros(1, dtype=torch.float32, device=est_tensor.device)

    for i in range(est_tensor.shape[0]):
        est_tensor_i = est_tensor[i, :lengths[i], :]
        target_tensor_i = target[i, :lengths[i], :]
        if est_tensor_i.shape[0] != target_tensor_i.shape[0]:
            # If the shapes do not match, truncate the longer tensor to match the shorter one
            min_length = min(est_tensor_i.shape[0], target_tensor_i.shape[0])
            est_tensor_i = est_tensor_i[:min_length, :]
            target_tensor_i = target_tensor_i[:min_length, :]
        assert est_tensor_i.shape == target_tensor_i.shape, "Estimated and target tensors must have the same shape."
        # Calculate the length of the vocal tract data
        est_tensor_i = est_tensor_i.to('cuda' if torch.cuda.is_available() else 'cpu')
        target_tensor_i = target_tensor_i.to('cuda' if torch.cuda.is_available() else 'cpu')
        loss += torch.mean((torch.norm(est_tensor_i - target_tensor_i, dim=-1)))  # Mean Euclidean distance
    loss /= est_tensor.shape[0]  # Average loss over the batch
    return loss


def get_f0(intrinsic_feature: torch.Tensor):
    # Calculate the fundamental frequency (F0) from intrinsic features.
    # shape of intrinsic_feature: (batch, time, 3)
    eps = 1e-8

    a_ta, a_ct, a_lc = intrinsic_feature[:, :, 0], intrinsic_feature[:, :, 1], intrinsic_feature[:, :, 2]
    elag = 0.2*(3.0*a_ct-a_ta)-0.2*a_lc
    length = 1.6*(1+elag)
    d_b = (a_ta * 0.4 +0.5*0.2)/(1+0.2*elag)
    d = d_b + (0.2 +0.5*0.2)/(1+0.2*elag)
    param = torch.tensor([[-0.5,-0.5,-0.5],[-0.35,0,-0.05],[0.5,0.4,1.0],[30.0, 1.39, 1.5],[4.4,17,6.5]], dtype=torch.float32, device=intrinsic_feature.device)
    sig_p = []
    for i in range(param.shape[1]):
        relu1 = F.relu((param[2,i].item()/param[0,i].item())*(elag - param[0,i].item()))
        relu2 = F.relu(param[3,i].item()*(torch.exp(param[4,i].item() * (elag - param[1,i].item())) -param[4,i].item() * (elag - param[1,i].item()) -1))
        sig_p.append((-relu1 + relu2))
    sig_p = torch.stack(sig_p, dim=-1)
    sig_p = torch.sum(sig_p, dim=-1)
    f0 = (1/2*length) * torch.sqrt((sig_p/1.04)*(1+(d_b*105*a_ta)/(d*sig_p+eps))+eps) 
    return f0 # shape: (batch, time)


# loss += torch.sum((est_f0_i - target_f0_i) ** 2 * mask) / torch.sum(mask + eps) # 마스크된 평균
def f0_loss(est_f0_feature, target_f0):
    """
    Calculate the loss between estimated and target fundamental frequency (F0).
    
    Parameters:
    est_f0 (torch.Tensor): Estimated F0 tensor.
    target_f0 (PackedSequence): Packed sequence of target F0 values.
    Returns:
    torch.Tensor: Loss value.
    """
    eps = 1e-8  # Small value to avoid division by zero
    target_f0, lengths = pad_packed_sequence(target_f0, batch_first=True)
    est_f0 = get_f0(est_f0_feature)
    
    # (batch, time)
    assert est_f0.shape[0] == target_f0.shape[0], "Estimated and target F0 tensors must have the same shape."
    loss = torch.zeros(1, dtype=torch.float32, device=est_f0.device)
    for i in range(est_f0.shape[0]):
        est_f0_i = est_f0[i, :lengths[i]]
        target_f0_i = target_f0[i, :lengths[i]]
        if est_f0_i.shape[0] != target_f0_i.shape[0]:
            # If the shapes do not match, truncate the longer tensor to match the shorter one
            min_length = min(est_f0_i.shape[0], target_f0_i.shape[0])
            est_f0_i = est_f0_i[:min_length]
            target_f0_i = target_f0_i[:min_length]
        assert est_f0_i.shape == target_f0_i.shape, "Estimated and target F0 tensors must have the same shape."
        mask = (target_f0_i > 0).float() # target_f0_i가 0보다 큰 경우에만 마스크 생성
        est_f0_i = est_f0_i.to('cuda' if torch.cuda.is_available() else 'cpu')
        target_f0_i = target_f0_i.to('cuda' if torch.cuda.is_available() else 'cpu')
        mask = mask.to(est_f0_i.device)
        est_f0_i = est_f0_i * mask  # Apply mask to estimated F0
        target_f0_i = target_f0_i * mask  # Apply mask to target F0
        # Calculate the loss
        loss += torch.sum((est_f0_i - target_f0_i) ** 2)/ torch.sum(mask+eps)  # Mean Squared Error
    return loss / est_f0.shape[0]  # Average loss over the batch


# def spl_corr_loss(est_spl_feature, target_spl):
#     """
#     Calculate the loss between estimated and target sound pressure level (SPL).
    
#     Parameters:
#     est_spl (torch.Tensor): Estimated SPL tensor.
#     target_spl (torch.Tensor): packed Target SPL tensor.
    
#     Returns:
#     torch.Tensor: Loss value.
#     """
#     eps = 1e-6  # Small value to avoid division by zero
#     target, lengths = pad_packed_sequence(target_spl, batch_first=True)
#     est_spl_feature = torch.log(est_spl_feature+eps)  # Convert estimated SPL to dB scale
    

#     # (batch, time)
#     assert est_spl_feature.shape[0] == target.shape[0], "Estimated and target SPL tensors must have the same shape."
#     corr = torch.zeros(1, dtype=torch.float32, device=est_spl_feature.device)
#     for i in range(est_spl_feature.shape[0]):
#         est_spl_feature_i = est_spl_feature[i, :lengths[i]]
#         target_i = target[i, :lengths[i]]
#         est_spl_feature_i = est_spl_feature_i.to('cuda' if torch.cuda.is_available() else 'cpu')
#         target_i = target_i.to('cuda' if torch.cuda.is_available() else 'cpu')
#         if est_spl_feature_i.shape[0] != target_i.shape[0]:
#             # If the shapes do not match, truncate the longer tensor to match the shorter one
#             min_length = min(est_spl_feature_i.shape[0], target_i.shape[0])
#             est_spl_feature_i = est_spl_feature_i[:min_length]
#             target_i = target_i[:min_length]
#         assert est_spl_feature_i.shape == target_i.shape, "Estimated and target SPL tensors must have the same shape."
#         # Calculate the correlation loss
#         est_spl_feature_i = est_spl_feature_i.to('cuda' if torch.cuda.is_available() else 'cpu')
#         target_i = target_i.to('cuda' if torch.cuda.is_available() else 'cpu')
#         corr += torch.sum(est_spl_feature_i * target_i,dim=-1) / (torch.norm(est_spl_feature_i, dim=-1) * torch.norm(target_i, dim=-1)+ eps)  # Cosine similarity
#     loss = 1/(eps+corr)
#     return loss

# def spl_std_loss(est_spl_feature, target_spl):
#     """
#     Calculate the standard deviation loss between estimated and target sound pressure level (SPL).
    
#     Parameters:
#     est_spl (torch.Tensor): Estimated SPL tensor.
#     target_spl (torch.Tensor): packed Target SPL tensor.
    
#     Returns:
#     torch.Tensor: Loss value.
#     """
#     eps = 1e-6  # Small value to avoid division by zero
#     target, lengths = pad_packed_sequence(target_spl, batch_first=True)
#     est_spl_feature = torch.log(est_spl_feature+eps)  # Convert estimated SPL to dB scale

#     # (batch, time)
#     assert est_spl_feature.shape[0] == target.shape[0], "Estimated and target SPL tensors must have the same shape."
#     std_loss = torch.zeros(1, dtype=torch.float32, device=est_spl_feature.device)
#     for i in range(est_spl_feature.shape[0]):
#         est_spl_feature_i = est_spl_feature[i, :lengths[i]]
#         target_i = target[i, :lengths[i]]
#         est_spl_feature_i = est_spl_feature_i.to('cuda' if torch.cuda.is_available() else 'cpu')
#         target_i = target_i.to('cuda' if torch.cuda.is_available() else 'cpu')
#         if est_spl_feature_i.shape[0] != target_i.shape[0]:
#             # If the shapes do not match, truncate the longer tensor to match the shorter one
#             min_length = min(est_spl_feature_i.shape[0], target_i.shape[0])
#             est_spl_feature_i = est_spl_feature_i[:min_length]
#             target_i = target_i[:min_length]
#         assert est_spl_feature_i.shape == target_i.shape, "Estimated and target SPL tensors must have the same shape."
#         # Calculate the standard deviation loss
#         std_loss += torch.std(est_spl_feature_i / target_i, dim=-1)/ (torch.norm(est_spl_feature_i, dim=-1) * torch.norm(target_i, dim=-1)+ eps)
#     return std_loss/ est_spl_feature.shape[0]  # Average loss over the batch


# def total_loss_with_dynamic_weighting(est, target,
#                                     initial_losses:dict = {'vt': None, 'f0': None, 'spl_corr': None, 'spl_std': None},
#                                     temperature=2.0):
#     """
#     Calculate the total loss between estimated and target tensors.
    
#     Parameters:
#     est (tuple): Tuple containing estimated tensors (vt, f0, spl).
#     target (tuple): Tuple containing target tensors (vt, f0, spl).
    
#     Returns:
#     torch.Tensor: Total loss value.
#     """
#     vt_loss_value = vt_loss(est[:,:,4:], target[0])
#     f0_loss_value = f0_loss(est[:,:,:3], target[1])
#     spl_corr_loss_value = spl_corr_loss(est[:,:,3], target[2])
#     spl_std_loss_value = spl_std_loss(est[:,:,3], target[2])
    

#     current_losses = {
#         'vt': vt_loss_value,
#         'f0': f0_loss_value,
#         'spl_corr': spl_corr_loss_value,
#         'spl_std': spl_std_loss_value
#     }

#     # 첫 에포크 (또는 초기화가 필요한 시점)에서 초기 손실 값 저장
#     if initial_losses['vt'] is None: # 혹은 특정 조건
#         for key in initial_losses:
#             initial_losses[key] = current_losses[key].detach() # detach해서 그래디언트 흐름 끊기

#     # 가중치 계산 (초기 손실이 설정된 이후에만)
#     if initial_losses['vt'] is not None:
#         weights = {}
#         sum_exp_term = 0.0
#         for key in initial_losses:
#             # 안전을 위해 작은 epsilon 더하기
#             relative_loss = current_losses[key] / (initial_losses[key] + 1e-8)
#             exp_term = torch.exp(relative_loss / temperature)
#             weights[key] = exp_term
#             sum_exp_term += exp_term

#         # 정규화된 가중치
#         for key in weights:
#             weights[key] = weights[key] / (sum_exp_term + 1e-8)
#     else:
#         # 초기화 전에는 균등 가중치 또는 다른 기본 가중치 사용
#         weights = {'vt': 0.25, 'f0': 0.25, 'spl_corr': 0.25, 'spl_std': 0.25}

#     total_loss_value = weights['vt'] * vt_loss_value + \
#                        weights['f0'] * f0_loss_value + \
#                        weights['spl_corr'] * spl_corr_loss_value + \
#                        weights['spl_std'] * spl_std_loss_value

#     return total_loss_value

def spl_mse_loss(est_spl_feature, target_spl):
    """
    Calculate the Mean Squared Error loss for SPL.
    """
    eps = 1e-8 # 작은 epsilon 값
    target, lengths = pad_packed_sequence(target_spl, batch_first=True)
    
    # est_spl_feature는 로그 스케일로 변환되었으므로, 타겟도 맞춰주거나
    # 아니면 로그 스케일 변환을 모델에서 처리하도록 하고 손실에서는 단순 MSE를 사용합니다.
    # 일단은 est_spl_feature가 이미 적절한 스케일이라고 가정하고, target도 유사한 스케일이라고 가정.
    # 만약 est_spl_feature가 선형 SPL 값이고, target_spl이 dB라면,
    # est_spl_feature = torch.log(est_spl_feature + eps)를 여기서 할 수도 있습니다.
    # 혹은 모델 출력단에서 log 스케일로 만들도록 처리하는 것이 더 일반적입니다.
    
    loss = torch.zeros(1, dtype=torch.float32, device=est_spl_feature.device)
    for i in range(est_spl_feature.shape[0]):
        est_spl_feature_i = est_spl_feature[i, :lengths[i]]
        target_i = target[i, :lengths[i]]
        
        min_length = min(est_spl_feature_i.shape[0], target_i.shape[0])
        est_spl_feature_i = est_spl_feature_i[:min_length]
        target_i = target_i[:min_length]
        
        # Ensure tensors are on the same device
        est_spl_feature_i = est_spl_feature_i.to('cuda' if torch.cuda.is_available() else 'cpu')
        target_i = target_i.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Calculate MSE loss
        loss += F.mse_loss(est_spl_feature_i, target_i, reduction='mean') # 각 시퀀스별 평균
        
    return loss / est_spl_feature.shape[0] # 배치 평균


# total_loss_with_dynamic_weighting 함수 수정
def total_loss_with_dynamic_weighting(est, target,
                                       initial_losses:dict = {'vt': None, 'f0': None, 'spl': None}, # spl_corr, spl_std 대신 spl 하나로
                                       temperature=40.0):
    vt_loss_value = vt_loss(est[:,:,4:], target[0])
    f0_loss_value = f0_loss(est[:,:,:3], target[1])
    spl_loss_value = spl_mse_loss(est[:,:,3], target[2]) # 새로운 spl_mse_loss 사용
    
    current_losses = {
        'vt': vt_loss_value,
        'f0': f0_loss_value,
        'spl': spl_loss_value # 키 이름도 변경
    }

    # 첫 에포크 (또는 초기화가 필요한 시점)에서 초기 손실 값 저장
    if initial_losses['vt'] is None:
        for key in initial_losses:
            initial_losses[key] = current_losses[key].detach()

    # 가중치 계산 (초기 손실이 설정된 이후에만)
    if initial_losses['vt'] is not None:
        weights = {}
        sum_exp_term = 0.0
        # 이 부분이 여전히 NaN을 유발할 수 있습니다.
        # current_losses[key] / (initial_losses[key] + 1e-8)
        # 만약 initial_losses[key]가 아주 커서 current_losses[key]가 상대적으로 작아지면,
        # relative_loss가 0에 가깝거나 아주 작아질 수 있고, exp(아주 작은 값)도 아주 작은 값이 됩니다.
        # 반대로 initial_losses[key]가 0에 가까운데 current_losses[key]가 크면 relative_loss가 매우 커지고,
        # exp(매우 큰 값)은 inf가 됩니다.
        for key in initial_losses:
            relative_loss = current_losses[key] / (initial_losses[key] + 1e-8) # 작은 epsilon 더하기
            
            # 여기서 relative_loss에 NaN/inf가 있는지 디버깅 출력 추가
            if torch.isnan(relative_loss).any() or torch.isinf(relative_loss).any():
                print(f"DEBUG: NaN/Inf in relative_loss for {key}. current: {current_losses[key].item()}, initial: {initial_losses[key].item()}")
                # raise ValueError(f"NaN/Inf in relative_loss for {key}")

            exp_term = torch.exp(relative_loss / temperature)
            
            # 여기서 exp_term에 NaN/inf가 있는지 디버깅 출력 추가
            if torch.isnan(exp_term).any() or torch.isinf(exp_term).any():
                print(f"DEBUG: NaN/Inf in exp_term for {key}. relative_loss: {relative_loss.item()}")
                # raise ValueError(f"NaN/Inf in exp_term for {key}")

            weights[key] = exp_term
            sum_exp_term += exp_term

        # 정규화된 가중치
        # sum_exp_term이 0에 가까워지면 문제가 될 수 있습니다. (모든 exp_term이 0에 수렴하는 경우)
        for key in weights:
            weights[key] = weights[key] / (sum_exp_term + 1e-8) # 정규화 시에도 epsilon
    else:
        weights = {'vt': 1/3, 'f0': 1/3, 'spl': 1/3} # 가중치 갯수 조정

    total_loss_value = weights['vt'] * vt_loss_value + \
                       weights['f0'] * f0_loss_value + \
                       weights['spl'] * spl_loss_value # spl_corr, spl_std 대신 spl 하나로
    
    return total_loss_value


# dataset = AudioDataset('Dataset', 'Dataset/npz', sample_rate=16000)  # Initialize the dataset with the data directory and CSV directory

# dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True, collate_fn = pack_collate_fn)  # Create a DataLoader for the dataset

# example_audio, labels = next(iter(dataloader))  # Get a batch of data from the DataLoader
# unpacked_audio = pad_packed_sequence(example_audio, batch_first=True)[0]  # Unpack the audio tensor
# audio_lengths = pad_packed_sequence(example_audio, batch_first=True)[1]  # Get the lengths of the audio sequences
# for i, length in enumerate(audio_lengths):
#     print(unpacked_audio[i, :length].shape)  # Unpack the audio tensor for each sequence length

# vt_image, f0s, spls = labels  # Unpack the labels
# print(f"Labels: {pad_packed_sequence(vt_image,batch_first=True)[0].shape}")  # Print the labels (inner_tract, outer_tract, f0, spl)

# spec = get_spectrogram(unpacked_audio[0].numpy(), sample_rate=16000)  # Get the spectrogram of the first audio in the batch
# print(f"Spectrogram shape: {spec.shape}")  # Print the shape of the spectrogram
# spec = spec.transpose(0, 1)  # Transpose to (frequency, time) format
# print(f"Spectrogram shape: {spec.shape}")  # Print the shape of the spectrogram



"""
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
