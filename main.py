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


from model import Model
from utils import AudioDataset, get_spectrogram, estimate_spectrogram_spl, estimate_fundamental_frequency, \
                                pack_collate_fn, get_f0, f0_loss, spl_mse_loss
import imageencoder as ie

class Config:
    def __init__(self):
        """
        Configuration class to hold hyperparameters and paths.
        """
        self.gpu = 0
        self.calc_gpu = 0
        self.epoch = 5
        self.use_amp = False
        self.model_dir = "models"
        self.model_name = "model.pth"
        self.load_model = True  # Set to True if you want to load a pre-trained model
        self.vt_threshold = 70  # Threshold for voice timbre accuracy
        self.f0_threshold = 10  # Threshold for fundamental frequency accuracy
        self.spl_threshold = 60  # Threshold for sound pressure level accuracy
        self.decoder_epoch = 20
        self.decoder_dir = "imageencoder"
        self.decoder_name = "image_encoder.pth".replace(".pth", f"_epoch{self.decoder_epoch}.pth")  # Name of the image decoder model file
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.decoder_dir):
            os.makedirs(self.decoder_dir)


class MainDiagnoser(nn.Module):
    def __init__(self, config:Config = Config()):
        super().__init__()
        self.config = config
        self.device = torch.device(f"cuda:{self.config.gpu}" if torch.cuda.is_available() else "cpu")
        self.model = Model().to(self.device)
        self.calc_device = torch.device("cuda:%d" % self.config.calc_gpu if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler(enabled=self.config.use_amp)
        self._load_model()
        self.image_decoder = ie.Decoder().to(self.device)  # Initialize the image decoder
        self.load_image_decoder()  # Load the image decoder model


    def _load_model(self):
        """
        Load the pre-trained model if specified in the configuration.
        """
        if self.config.load_model:
            model_path = os.path.join(self.config.model_dir, self.config.model_name.replace(".pth", f"_epoch_{self.config.epoch}.pth")) # Adjust the path to include epoch
            if os.path.exists(model_path):
                torch.load(model_path, weights_only=True, map_location=self.device)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                print(f"Model loaded from {model_path}")
            else:
                print(f"Model file {model_path} does not exist. Starting with a new model.")
        else:
            print("Not loading any pre-trained model.")

    
    def load_image_decoder(self):
        """
        Load the image decoder model from the specified path.
        
        Args:
            decoder_path (str): Path to the image decoder model file.
        """
        decoder_path = os.path.join(self.config.decoder_dir, self.config.decoder_name)
        if os.path.exists(decoder_path):
            self.image_decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))
            self.image_decoder.to(self.device)
            print(f"Image decoder loaded from {decoder_path}")
        else:
            print(f"Image decoder file {decoder_path} does not exist. Starting with a new decoder.")
        self.image_decoder.eval()  # Set the decoder to evaluation mode
    

    def inference(self, audio_data):
        """
        Perform inference on the input audio data.
        
        Args:
            audio_data (torch.Tensor): Input audio data tensor.
        
        Returns:
            torch.Tensor: Output from the model.
        """
        if audio_data.dim() == 1:
            audio_data = audio_data.unsqueeze(0)
        audio_data = pack_sequence(audio_data, enforce_sorted=False)
        self.model.eval()
        with torch.no_grad():
            audio_data = audio_data.to(self.device)
            output = self.model(audio_data)
        

        return output
    
    # Calculate the accuracy of the model's predictions.
    # vt_acc: cosine similarity of vt features for voice timbre and pronounciation
    # f0_acc: accuracy of fundamental frequency estimation within 1/12 octave
    # spl_acc: cosine similarity for sound pressure level estimation(for dinamic range or onset detection)
    def accuracy_function(self, est, target, reference_f0, user_f0):
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
            # f0_target = target[1]
            spl_target = target[2]
            vt_tensor, vt_lengths = pad_packed_sequence(vt_target, batch_first=True)
            vt_tensor = vt_tensor.reshape(vt_tensor.shape[0], vt_tensor.shape[1], -1)
            # f0_tensor, f0_lengths = pad_packed_sequence(f0_target, batch_first=True)
            # f0_tensor = f0_tensor.to(self.calc_device) # Move to calc_device
            spl_tensor, spl_lengths = pad_packed_sequence(spl_target, batch_first=True)
            spl_tensor = spl_tensor.to(self.calc_device)  # Move to calc_device
        else:
            vt_tensor = target[:,:,4:]  # Assuming vt_tensor is already in the correct shape
            f0_tensor = reference_f0  # Assuming f0_tensor is already in the correct shape
            spl_tensor = target[:,:,3]  # Assuming spl_tensor is already in the correct shape
            vt_lengths = torch.tensor([vt_tensor.shape[1]], dtype=torch.int64, device=self.calc_device)
            f0_lengths = torch.tensor([f0_tensor.shape[1]], dtype=torch.int64, device=self.calc_device)
            spl_lengths = torch.tensor([spl_tensor.shape[1]], dtype=torch.int64, device=self.calc_device)
        vt_est = est[:, :, 4:]
        f0_est = user_f0
        spl_est = est[:, :, 3]

        # Ensure the shapes match
        if vt_tensor.shape!= vt_est.shape:
            min_length = min(vt_tensor.shape[1], vt_est.shape[1])
            if torch.max(vt_lengths).item() > min_length:
                vt_lengths = torch.clamp(vt_lengths, max=min_length)
            vt_tensor = vt_tensor[:, :min_length, :]
            vt_est = vt_est[:, :min_length, :]
        vt_est = vt_est.to(self.calc_device)  # Move to calc_device
        vt_tensor = vt_tensor.to(self.calc_device)  # Move to calc_device
        if f0_est.dim() == 3:
            f0_est = get_f0(f0_est).to(self.calc_device)  # Move to calc_device  # Move to calc_device
        else:
            f0_est = f0_est.to(self.calc_device)  # Move to calc_device
        if f0_tensor.dim() == 3:
            f0_tensor = get_f0(f0_tensor).to(self.calc_device)  # Move to calc_device
        else:
            f0_tensor = f0_tensor.to(self.calc_device)
        if f0_tensor.shape != f0_est.shape:
            min_length = min(f0_tensor.shape[1], f0_est.shape[1])
            if torch.max(f0_lengths).item() > min_length:
                f0_lengths = torch.clamp(f0_lengths, max=min_length)
            f0_tensor = f0_tensor[:, :min_length]
            f0_est = f0_est[:, :min_length]
        f0_est = f0_est.to(self.calc_device)  # Move to calc_device
        f0_tensor = f0_tensor.to(self.calc_device)  # Move to calc_device
        if spl_tensor.shape != spl_est.shape:
            min_length = min(spl_tensor.shape[1], spl_est.shape[1])
            if torch.max(spl_lengths).item() > min_length:
                spl_lengths = torch.clamp(spl_lengths, max=min_length)
            spl_tensor = spl_tensor[:, :min_length]
            spl_est = spl_est[:, :min_length]
        print(f"f0_tensor shape: {f0_tensor.shape}, f0_est shape: {f0_est.shape}")
        #normalize spl_est and spl_tensor
        if spl_est.shape[1] == 0 or spl_tensor.shape[1] == 0:
            spl_est = torch.ones((spl_est.shape[0], 1), device=self.calc_device)  # Avoid division by zero
            spl_tensor = torch.ones((spl_tensor.shape[0], 1), device=self.calc_device)
            
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
            vt_acc = torch.cosine_similarity(vt_tensor_i, vt_est_i, dim=-1, eps=1e-6) # outputs are bigger than 0, so cosine similarity is bound to [0, 1]
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
            f0_acc_count = f0_est_i[torch.abs(torch.log2(f0_est_i+1e-6) - torch.log2(f0_tensor_i+1e-8)) < 1/12].shape[0]
            f0_acc = f0_acc_count / f0_tensor_i.shape[0] if f0_tensor_i.shape[0] > 0 else 0
            f0_acc_list.append(f0_acc)
            spl_tensor_i = spl_tensor[i,:spl_lengths[i]].to(self.calc_device)
            spl_est_i = spl_est[i,:spl_lengths[i]].to(self.calc_device)
            spl_acc = torch.cosine_similarity(spl_tensor_i, spl_est_i, dim=-1, eps=1e-8)
            spl_acc_list.append(spl_acc)

        # Calculate average accuracy across the batch
        if len(vt_acc_list) == 0:
            vt_acc = 0.0
        else:
            vt_acc = 100*sum(vt_acc_list) / len(vt_acc_list)
        if len(f0_acc_list) == 0:
            f0_acc = 0.0
        else:
            f0_acc = 100*sum(f0_acc_list) / len(f0_acc_list)
        if len(spl_acc_list) == 0:
            spl_acc = 0.0
        else:
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
        if reference_audio.dim == 1 and reference_audio.shape[0] != user_audio.shape[0]:
            min_length = min(reference_audio.shape[0], user_audio.shape[0])
            reference_audio = reference_audio[:min_length]
            user_audio = user_audio[:min_length]
            # add a batch dimension
            reference_audio = reference_audio.unsqueeze(0) if reference_audio.ndim == 1 else reference_audio
            user_audio = user_audio.unsqueeze(0) if user_audio.ndim == 1 else user_audio
        reference_features = self.inference(reference_audio)
        user_features = self.inference(user_audio)
      
        reference_f0 = reference_f0.unsqueeze(0) if reference_f0.ndim == 1 else reference_f0  # Ensure reference_f0 has a batch dimension
        user_f0 = user_f0.unsqueeze(0) if user_f0.ndim == 1 else user_f0  # Ensure user_f0 has a batch dimension
        vt_diagnosis_list = []
        f0_diagnosis_list = []
        spl_diagnosis_list = []
        chunk_size = int(23.18 * 20)  # about 20 seconds in output results
        if reference_features.shape[1] >  chunk_size:
            for i in range(0, reference_features.shape[1], chunk_size):
                if i + chunk_size < reference_features.shape[1]:
                    reference_chunk = reference_features[:, i:i + chunk_size, :]
                    user_chunk = user_features[:, i:i + chunk_size, :]
                    reference_f0_chunk = reference_f0[:, i:i + chunk_size]
                    user_f0_chunk = user_f0[:, i:i + chunk_size]
                else:
                    reference_chunk = reference_features[:, i:, :]
                    user_chunk = user_features[:, i:, :]
                    reference_f0_chunk = reference_f0[:, i:]
                    user_f0_chunk = user_f0[:, i:]
                vt_acc, f0_acc, spl_acc, _ = self.accuracy_function(user_chunk, reference_chunk, reference_f0_chunk, user_f0_chunk)
                print(f"Chunk {i//chunk_size}: vt_acc={vt_acc:.2f}, f0_acc={f0_acc:.2f}, spl_acc={spl_acc:.2f}")
                vt_diagnosis_list.append(1 if vt_acc < self.config.vt_threshold else 0)
                f0_diagnosis_list.append(1 if f0_acc < self.config.f0_threshold else 0)
                spl_diagnosis_list.append(1 if spl_acc < self.config.spl_threshold else 0)
        else:
            vt_acc, f0_acc, spl_acc, _ = self.accuracy_function(user_features, reference_features, reference_f0, user_f0)
            vt_diagnosis_list.append(1 if vt_acc < self.config.vt_threshold else 0)
            f0_diagnosis_list.append(1 if f0_acc < self.config.f0_threshold else 0)
            spl_diagnosis_list.append(1 if spl_acc < self.config.spl_threshold else 0)
        
        return reference_features, user_features, vt_diagnosis_list, f0_diagnosis_list, spl_diagnosis_list

                
if __name__ == "__main__":
    config = Config()
    diagnoser = MainDiagnoser(config)
    
    # Example usage with dummy data
    reference_audio, sr = torchaudio.load("letitbe.wav")
    if reference_audio.dim() > 1 and reference_audio.shape[0] > 1:
        reference_audio = reference_audio.mean(dim=0, keepdim=False)
    elif reference_audio.shape[0] == 1:
        reference_audio = reference_audio.squeeze(0)
    if sr != 16000:
        reference_audio = librosa.resample(reference_audio.numpy(), orig_sr=sr, target_sr=16000)
        reference_audio = torch.tensor(reference_audio, dtype=torch.float32)
    
    # Dummy sound pressure level for reference audio
    reference_f0 = estimate_fundamental_frequency(reference_audio.numpy())
    reference_f0 = torch.tensor(reference_f0, dtype=torch.float32)
    reference_spl = torch.tensor([70.0])  # Dummy sound pressure level

    user_audio, sr = torchaudio.load("record_20250618_112441_69_72.wav")
    if user_audio.dim() > 1 and user_audio.shape[0] > 1:
        user_audio = user_audio.mean(dim=0, keepdim=False)
    elif user_audio.shape[0] == 1:
        user_audio = user_audio.squeeze(0)
    if sr != 16000:
        user_audio = librosa.resample(user_audio.numpy(), orig_sr=sr, target_sr=16000)
        user_audio = torch.tensor(user_audio, dtype=torch.float32)
    
    user_f0 = estimate_fundamental_frequency(user_audio.numpy())
    user_f0 = torch.tensor(user_f0, dtype=torch.float32)
    user_spl = torch.tensor([70.0])  # Dummy sound pressure level

    
    
    reference_features, user_features, vt_diagnosis, f0_diagnosis, spl_diagnosis = diagnoser.diagnose(reference_audio, reference_f0,
                                                                    reference_spl, user_audio,
                                                                    user_f0, user_spl)
    
    print("Voice Timbre Diagnosis:", vt_diagnosis)
    print("Fundamental Frequency Diagnosis:", f0_diagnosis)
    print("Sound Pressure Level Diagnosis:", spl_diagnosis)
    # plot generated vt-image
    reference_feature_latent = reference_features[:, :, 4:]
    user_features_latent = user_features[:, :, 4:]
    reference_feature_reshape = reference_feature_latent.reshape(reference_feature_latent.shape[0], reference_feature_latent.shape[1], 6, 6)
    user_feature_reshape = user_features_latent.reshape(user_features_latent.shape[0], user_features_latent.shape[1], 6, 6)
    if reference_feature_reshape.shape != user_feature_reshape.shape:
        print(f"Reference feature shape: {reference_feature_reshape.shape}, User feature shape: {user_feature_reshape.shape}")
        min_length = min(reference_feature_reshape.shape[1], user_feature_reshape.shape[1])
        if reference_feature_reshape.shape[1] > min_length:
            reference_feature_reshape = reference_feature_reshape[:, :min_length, :, :]
            user_feature_reshape = user_feature_reshape[:, :min_length, :, :]
    
    assert reference_feature_reshape.shape == user_feature_reshape.shape, "Reference and user feature shapes do not match"
    
    with torch.no_grad():
        if reference_feature_reshape.shape[1] > 512:
            reference_feature_list = []
            user_features_list = []
            for i in range(0, reference_feature_reshape.shape[1], 512):
                if i + 512 < reference_feature_reshape.shape[1]:
                    reference_feature_chunk = reference_feature_reshape[0, i:i + 512, :, :].unsqueeze(1)  # Take the first batch and first channel
                    user_feature_chunk = user_feature_reshape[0, i:i + 512, :, :].unsqueeze(1)  # Take the first batch and first channel
                else:
                    reference_feature_chunk = reference_feature_reshape[0, i:, :, :].unsqueeze(1)  # Take the first batch and first channel
                    user_feature_chunk = user_feature_reshape[0, i:, :, :].unsqueeze(1)  # Take the first batch and first channel
                reference_feature_chunk = diagnoser.image_decoder(reference_feature_chunk)
                user_feature_chunk = diagnoser.image_decoder(user_feature_chunk)
                reference_feature_chunk = reference_feature_chunk.squeeze(1).cpu().detach().numpy()  # Remove the channel dimension
                user_feature_chunk = user_feature_chunk.squeeze(1).cpu().detach().numpy()
                reference_feature_list.append(reference_feature_chunk)
                user_features_list.append(user_feature_chunk)
            reference_feature = np.concatenate(reference_feature_list, axis=0)
            user_feature = np.concatenate(user_features_list, axis=0)
        else:
            reference_feature = reference_feature_reshape[0, :, :, :].unsqueeze(1)  # Take the first batch and first channel
            user_feature = user_feature_reshape[0, :, :, :].unsqueeze(1)  # Take the first batch and first channel
            reference_feature = diagnoser.image_decoder(reference_feature)
            user_feature = diagnoser.image_decoder(user_feature)
            reference_feature = reference_feature.squeeze(1).cpu().detach().numpy()  # Remove the channel dimension
            user_feature = user_feature.squeeze(1).cpu().detach().numpy()
    

    # Plot the generated VT image
    vt_image1 = user_feature[1300,:,:]
    plt.figure(figsize=(6, 6))
    plt.imshow(vt_image1, cmap='gray')
    plt.title("Generated VT Image")
    plt.axis('off')
    plt.show()
    plt.gca().invert_xaxis()  # Invert x-axis for correct orientation
    plt.gca().invert_yaxis()  # Invert y-axis for correct orientation
    # Save image
    plt.savefig("user_generated_image_encoded.png")
    vt_image2 = reference_feature[1300,:,:]
    plt.figure(figsize=(6, 6))
    plt.imshow(vt_image2, cmap='gray')
    plt.title("Reference VT Image")
    plt.axis('off')
    plt.show()
    plt.gca().invert_xaxis()  # Invert x-axis for correct orientation
    plt.gca().invert_yaxis()  # Invert y-axis for correct orientation
    # Save image
    plt.savefig("reference_generated_image_encoded.png")

    # save numpy array to video file
    reference_feature = np.flip(reference_feature, axis=1)  # Flip the feature map for correct orientation
    user_feature = np.flip(user_feature, axis=1)  # Flip the feature map for correct orientation
    reference_feature = np.flip(reference_feature, axis=2)  # Flip the feature map for correct orientation
    user_feature = np.flip(user_feature, axis=2)  # Flip the feature map
    
    # Convert to numpy array with shape (frames, height, width, channels)
    reference_video_np = np.zeros((reference_feature.shape[0], reference_feature.shape[1], reference_feature.shape[2], 3), dtype=np.uint8)
    user_video_np = np.zeros((user_feature.shape[0], user_feature.shape[1], user_feature.shape[2], 3), dtype=np.uint8)
    
    reference_video_np[:, :, :, 0] = reference_feature[:, :, :]
    reference_video_np[:, :, :, 1] = reference_feature[:, :, :]
    reference_video_np[:, :, :, 2] = reference_feature[:, :, :]
    user_video_np[:, :, :, 0] = user_feature[:, :, :]
    user_video_np[:, :, :, 1] = user_feature[:, :, :]
    user_video_np[:, :, :, 2] = user_feature[:, :, :]
    reference_video_np = reference_video_np.astype(np.uint8)
    user_video_np = user_video_np.astype(np.uint8)
    # Save as video
    diagnoser.image_decoder.numpy_array_to_video_ffmpeg(reference_video_np, "reference_vt_video.mp4")
    diagnoser.image_decoder.numpy_array_to_video_ffmpeg(user_video_np, "user_vt_video.mp4")




        