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

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pad_packed_sequence, PackedSequence
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import utils
import model
import letalker as lt
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from Trainer import accuracy_
from model import Model
from utils import AudioDataset, get_spectrogram, estimate_spectrogram_spl, estimate_fundamental_frequency, \
                                pack_collate_fn, vt_loss, get_f0, f0_loss, spl_mse_loss


class Config:
    def __init__(self):
        """
        Configuration class to hold hyperparameters and paths.
        """
        self.gpu = 0
        self.calc_gpu = 0
        self.use_amp = False
        self.model_dir = "models"
        self.model_name = "model.pth"
        self.load_model = True  # Set to True if you want to load a pre-trained model
        self.vt_threshold = 70  # Threshold for voice timbre accuracy
        self.f0_threshold = 0  # Threshold for fundamental frequency accuracy
        self.spl_threshold = 60  # Threshold for sound pressure level accuracy
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)


class MainDiagnoser(nn.Module):
    def __init__(self, config:Config = Config()):
        super().__init__()
        self.config = config
        self.device = torch.device(f"cuda:{self.config.gpu}" if torch.cuda.is_available() else "cpu")
        self.model = Model().to(self.device)
        self.scaler = GradScaler(enabled=self.config.use_amp)
        self._load_model()

    def _load_model(self):
        """
        Load the pre-trained model if specified in the configuration.
        """
        if self.config.load_model:
            model_path = os.path.join(self.config.model_dir, self.config.model_name)
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device),weight_only=True)
                print(f"Model loaded from {model_path}")
            else:
                print(f"Model file {model_path} does not exist. Starting with a new model.")
        else:
            print("Not loading any pre-trained model.")
    

    def inference(self, audio_data):
        """
        Perform inference on the input audio data.
        
        Args:
            audio_data (torch.Tensor): Input audio data tensor.
        
        Returns:
            torch.Tensor: Output from the model.
        """
        self.model.eval()
        with torch.no_grad():
            audio_data = audio_data.to(self.device)
            output = self.model(audio_data)
        

        return output
    
    # Calculate the accuracy of the model's predictions.
    # vt_acc: cosine similarity of vt features for voice timbre and pronounciation
    # f0_acc: accuracy of fundamental frequency estimation within 1/12 octave
    # spl_acc: cosine similarity for sound pressure level estimation(for dinamic range or onset detection)
    def accuracy_function(self, est, target):
        """
        Calculate the accuracy of the model's predictions.
        
        Args:
            est: Model outputs.
            target: Ground truth targets.
        
        Returns:
            Accuracy value.
        """
        eps = 1e-8
        if isinstance(target, tuple):
            vt_target = target[0]
            f0_target = target[1]
            spl_target = target[2]
            vt_tensor, vt_lengths = pad_packed_sequence(vt_target, batch_first=True)
            vt_tensor = vt_tensor.reshape(vt_tensor.shape[0], vt_tensor.shape[1], -1)
            f0_tensor, f0_lengths = pad_packed_sequence(f0_target, batch_first=True)
            f0_tensor = f0_tensor.to(self.calc_device) # Move to calc_device
            spl_tensor, spl_lengths = pad_packed_sequence(spl_target, batch_first=True)
            spl_tensor = spl_tensor.to(self.calc_device)  # Move to calc_device
        else:
            vt_tensor = target[:,:,4:]  # Assuming vt_tensor is already in the correct shape
            f0_tensor = target[:,:,:3]  # Assuming f0_tensor is already in the correct shape
            spl_tensor = target[:,:,3]  # Assuming spl_tensor is already in the correct shape
            vt_lengths = torch.tensor([vt_tensor.shape[1]] * vt_tensor.shape[0], dtype=torch.int64, device=self.calc_device)
            f0_lengths = torch.tensor([f0_tensor.shape[1]] * f0_tensor.shape[0], dtype=torch.int64, device=self.calc_device)
            spl_lengths = torch.tensor([spl_tensor.shape[1]] * spl_tensor.shape[0], dtype=torch.int64, device=self.calc_device)
        vt_est = est[:, :, 4:]
        f0_est = est[:, :, :3]
        spl_est = est[:, :, 3]

        # Ensure the shapes match
        if vt_tensor.shape!= vt_est.shape:
            min_length = min(vt_tensor.shape[1], vt_est.shape[1])
            if torch.max(vt_lengths) > min_length:
                vt_lengths = torch.clamp(vt_lengths, max=min_length)
            vt_tensor = vt_tensor[:, :min_length, :]
            vt_est = vt_est[:, :min_length, :]
        vt_est = vt_est.to(self.calc_device)  # Move to calc_device
        vt_tensor = vt_tensor.to(self.calc_device)  # Move to calc_device
        f0_est = get_f0(f0_est).to(self.calc_device)  # Move to calc_device  # Move to calc_device
        if f0_tensor.shape != f0_est.shape:
            min_length = min(f0_tensor.shape[1], f0_est.shape[1])
            if torch.max(f0_lengths) > min_length:
                f0_lengths = torch.clamp(f0_lengths, max=min_length)
            f0_tensor = f0_tensor[:, :min_length]
            f0_est = f0_est[:, :min_length]
        f0_est = f0_est.to(self.calc_device)  # Move to calc_device
        f0_tensor = f0_tensor.to(self.calc_device)  # Move to calc_device
        if spl_tensor.shape != spl_est.shape:
            min_length = min(spl_tensor.shape[1], spl_est.shape[1])
            if torch.max(spl_lengths) > min_length:
                spl_lengths = torch.clamp(spl_lengths, max=min_length)
            spl_tensor = spl_tensor[:, :min_length]
            spl_est = spl_est[:, :min_length]
        
        #normalize spl_est and spl_tensor
        spl_est = torch.log(spl_est + 1e-8)  # Logarithm for SPL estimation
        spl_est -= torch.max(spl_est, dim=-1, keepdim=True).values  # Normalize to zero mean
        spl_est = spl_est/(-torch.min(spl_est, dim=1, keepdim=True).values + eps)
        spl_tensor -= torch.max(spl_tensor, dim=-1, keepdim=True).values  # Normalize to zero mean
        spl_tensor = spl_tensor/(-torch.min(spl_tensor, dim=1, keepdim=True).values + eps)
        spl_est = spl_est.to(self.calc_device)  # Move to calc_device
        spl_tensor = spl_tensor.to(self.calc_device)  # Move to calc_device

        # est and reference tensors should have the same size
        assert vt_tensor.shape == vt_est.shape, "Batch size mismatch between target and estimate for vt"
        assert f0_tensor.shape == f0_est.shape, "Batch size mismatch between target and estimate for f0"
        assert spl_tensor.shape == spl_est.shape, "Batch size mismatch between target and estimate for spl"
        vt_acc_list = []
        f0_acc_list = []
        spl_acc_list = []

        # Calculate accuracy for each batch
        for i in range(vt_tensor.shape[0]):
            vt_tensor_i = vt_tensor[i,:vt_lengths[i],:].to(self.calc_device)
            vt_est_i = vt_est[i,:vt_lengths[i],:].to(self.calc_device)
            vt_acc = torch.cosine_similarity(vt_tensor_i, vt_est_i, dim=-1, eps=1e-8) # outputs are bigger than 0, so cosine similarity is bound to [0, 1]
            vt_acc = torch.mean(vt_acc,dim=-1)
            vt_acc_list.append(vt_acc)
            f0_tensor_i = f0_tensor[i,:f0_lengths[i]].to(self.calc_device)
            f0_est_i = f0_est[i,:f0_lengths[i]].to(self.calc_device)
            f0_est_i = f0_est_i[f0_tensor_i > eps]
            f0_tensor_i = f0_tensor_i[f0_tensor_i > eps]
            assert f0_tensor_i.shape == f0_est_i.shape, "Shape mismatch between target and estimate for f0"
            if f0_tensor_i.shape[0] == 0:
                print(f"Skipping batch {i} due to empty f0_tensor")
                continue
            f0_acc_count = f0_est_i[torch.abs(torch.log2(f0_est_i+1e-8) - torch.log2(f0_tensor_i+1e-8)) < 1/12].shape[0]
            f0_acc = f0_acc_count / f0_tensor_i.shape[0] if f0_tensor_i.shape[0] > 0 else 0
            f0_acc_list.append(f0_acc)
            spl_tensor_i = spl_tensor[i,:spl_lengths[i]].to(self.calc_device)
            spl_est_i = spl_est[i,:spl_lengths[i]].to(self.calc_device)
            spl_acc = torch.cosine_similarity(spl_tensor_i, spl_est_i, dim=-1, eps=1e-8)
            spl_acc_list.append(spl_acc)

        # Calculate average accuracy across the batch
        vt_acc = 100*sum(vt_acc_list) / len(vt_acc_list)
        f0_acc = 100*sum(f0_acc_list) / len(f0_acc_list)
        spl_acc = 100*sum(spl_acc_list) / len(spl_acc_list)
        
        # Calculate total accuracy as the average of the three accuracies
        total_acc = (vt_acc + f0_acc + spl_acc) / 3.

        return vt_acc, f0_acc, spl_acc, total_acc
    
    # 전처리 때 산출한 f0, spl은 인수로 추가해두었으나 현재 사용하지 않음. 추후 수정 가능성 있음.    
    def diagnose(self, reference_audio:torch.tensor, reference_f0:torch.tensor,
                 reference_spl:torch.tensor,
                 user_audio:torch.tensor, user_f0:torch.tensor,
                 user_spl:torch.tensor):
        """
        Diagnose two audio datas by computing the accuracy of voice timbre, fundamental frequency, and sound pressure level.
        
        Args:
            reference_audio, user_audio(torch.tensor): Audio data tensors for reference and user.
            reference_f0, user_f0(torch.tensor): Fundamental frequency tensors for reference and user (not currently used).
            reference_spl, user_spl(torch.tensor): Sound pressure level tensors for reference and user (not currently used).
        
        Returns:
            vt_diagnosis_list (list): List of voice timbre diagnosis results within 30 sec chunks.
            f0_diagnosis_list (list): List of fundamental frequency diagnosis results within 30 sec chunks.
            spl_diagnosis_list (list): List of sound pressure level diagnosis results within 30 sec chunks.
        """
        if reference_audio.shape[0] != user_audio.shape[0]:
            min_length = min(reference_audio.shape[0], user_audio.shape[0])
            reference_audio = reference_audio[:min_length]
            user_audio = user_audio[:min_length]
            # add a batch dimension
            reference_audio = reference_audio.unsqueeze(0) if reference_audio.ndim == 1 else reference_audio
            user_audio = user_audio.unsqueeze(0) if user_audio.ndim == 1 else user_audio
        reference_features = self.inference(reference_audio)
        user_features = self.inference(user_audio)
        
        vt_diagnosis_list = []
        f0_diagnosis_list = []
        spl_diagnosis_list = []
        chunk_size = int(23.18 * 30)  # about 30 seconds in output results
        if reference_features.shape[1] >  chunk_size:
            for i in range(0, reference_features.shape[1], chunk_size):
                if i + chunk_size < reference_features.shape[1]:
                    reference_chunk = reference_features[:, i:i + chunk_size, :]
                    user_chunk = user_features[:, i:i + chunk_size, :]
                else:
                    reference_chunk = reference_features[:, i:, :]
                    user_chunk = user_features[:, i:, :]
                vt_acc, f0_acc, spl_acc, _ = self.accuracy_function(user_chunk, reference_chunk)
                vt_diagnosis_list.append(1 if vt_acc < self.config.vt_threshold else 0)
                f0_diagnosis_list.append(1 if f0_acc < self.config.f0_threshold else 0)
                spl_diagnosis_list.append(1 if spl_acc < self.config.spl_threshold else 0)
        else:
            vt_acc, f0_acc, spl_acc, _ = self.accuracy_function(user_features, reference_features)
            vt_diagnosis_list.append(1 if vt_acc < self.config.vt_threshold else 0)
            f0_diagnosis_list.append(1 if f0_acc < self.config.f0_threshold else 0)
            spl_diagnosis_list.append(1 if spl_acc < self.config.spl_threshold else 0)
        
        return vt_diagnosis_list, f0_diagnosis_list, spl_diagnosis_list

                


        