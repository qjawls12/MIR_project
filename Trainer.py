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
                                total_loss_with_dynamic_weighting, pack_collate_fn


class Config:
    def __init__(self):
        """
        Configuration class to hold hyperparameters and paths.
        """
        self.gpu = 0
        self.calc_gpu = 0
        self.use_amp = False
        self.batch_size = 8
        self.learning_rate = 1e-10
        self.num_epochs = 50
        self.save_interval = 1
        self.model_dir = "models"
        self.model_name = "model.pth"
        self.epoch = 0
        self.load_model = False
        self.train_data_path = "Dataset"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)


class Trainer:
    def __init__(self, config:Config = Config()):
        """
        Constructor for the Trainer class.

        Initializes the trainer object with default values for the hyperparameters and data loaders.
        """
        self.config = config
        self.use_amp = self.config.use_amp
        if self.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        self.device = torch.device("cuda:%d" % self.config.gpu if torch.cuda.is_available() else "cpu")
        self.calc_device = torch.device("cuda:%d" % self.config.calc_gpu if torch.cuda.is_available() else "cpu")
        self._load_data()
        self._load_model()

    def _load_data(self):
        """
        Load the dataset and create data loaders.
        """
        self.dataset = AudioDataset(self.config.train_data_path, 'Dataset/npz_encoded', sample_rate=16000)
        train_size = int(0.9 * len(self.dataset))
        val_size = len(self.dataset) - train_size 
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=pack_collate_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=pack_collate_fn)

    def _load_model(self):
        """
        Load the model architecture.
        """
        self.model = Model()
        self.model.to(self.device)
        if self.config.load_model:
            model_path = os.path.join(self.config.model_dir, self.config.model_name.replace(".pth", f"_epoch_{self.config.epoch}.pth"))
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Model loaded from {model_path}")
            else:
                print(f"Model file {model_path} does not exist. Starting with a new model.")


    def train(self):
        """
        Train the model.
        """
        torch.cuda.empty_cache()
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)


        loss_history = []
        for epoch in tqdm(range(self.config.num_epochs), desc="Training Epochs", leave=False):
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}", leave=False):
                idx = 0
                inputs, targets = batch
                inputs = inputs.to(self.calc_device)
                vt_image, f0s, spls = targets
                if isinstance(f0s, torch.nn.utils.rnn.PackedSequence):
                    f0s = torch.nn.utils.rnn.PackedSequence(
                    f0s.data.to(self.calc_device),
                    f0s.batch_sizes,
                    f0s.sorted_indices.to(self.calc_device),
                    f0s.unsorted_indices.to(self.calc_device)
                    )
                else:
                    # 일반 텐서라면 그냥 to(device)
                    f0s = f0s.to(self.calc_device)
                if isinstance(spls, torch.nn.utils.rnn.PackedSequence):
                    spls = torch.nn.utils.rnn.PackedSequence(
                    spls.data.to(self.calc_device),
                    spls.batch_sizes,
                    spls.sorted_indices.to(self.calc_device),
                    spls.unsorted_indices.to(self.calc_device)
                    )
                else:
                    # 일반 텐서라면 그냥 to(device)
                    spls = spls.to(self.calc_device)
                # Move vt_image, f0s, spls to calc_device
                if isinstance(vt_image, torch.nn.utils.rnn.PackedSequence):
                    vt_image = torch.nn.utils.rnn.PackedSequence(
                        vt_image.data.to(self.calc_device),
                        vt_image.batch_sizes,
                        vt_image.sorted_indices.to(self.calc_device),
                        vt_image.unsorted_indices.to(self.calc_device)
                    )
                else:
                    # 일반 텐서라면 그냥 to(device)
                    vt_image = vt_image.to(self.calc_device)
                targets_on_device = (vt_image, f0s, spls)

                self.optimizer.zero_grad()
                with autocast(enabled=self.use_amp, device_type='cuda', dtype=torch.float16 if self.use_amp else torch.float32):
                    outputs = self.model(inputs)
                    loss = total_loss_with_dynamic_weighting(outputs, targets_on_device)
                    # train 메서드 내부, loss 계산 직후에 추가
                    if torch.isnan(loss).any():
                        print(f"NAN loss detected at Epoch {epoch+1}, Batch {idx+1}!")
                        print(f"Outputs device: {outputs.device}, dtype: {outputs.dtype}")
                        print(f"Targets device: {targets_on_device[0].data.device}, dtype: {targets_on_device[0].data.dtype}")
                        # 여기서 outputs와 targets_on_device의 일부를 출력하여 nan의 원인을 추적할 수 있습니다.
                        # 예: print(outputs[0, :10])
                        # 예: print(targets_on_device[0][0, :10])
                        # 예: 모델의 중간 레이어 출력에 hook을 걸어 nan/inf 발생 위치 확인 (고급 디버깅)
                        # raise ValueError("Loss became NaN. Aborting training.")
                # --- autocast 블록 끝 ---

                # --- AMP 사용 시 기울기 스케일링 및 백워드 로직 ---
                if self.use_amp:
                    # 1. 손실을 스케일링하고 backward 호출
                    #    이때 _scale 값이 설정됩니다.
                    self.scaler.scale(loss).backward() 
                
                    # 2. unscale_()은 clip_grad_norm_ 전에 필요
                    self.scaler.unscale_(self.optimizer) # 이 라인이 에러를 발생시킨 부분
                
                    # 3. 기울기 클리핑 (nan 문제 해결에도 도움)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.01) # max_norm 값 조정 가능
                
                    # 4. 옵티마이저 스텝 및 스케일러 업데이트
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # AMP를 사용하지 않을 경우 일반적인 backward() 및 step()
                    loss.backward()
                    # AMP 미사용 시에도 기울기 클리핑 적용 가능
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.01)
                    self.optimizer.step()
                # --- AMP 로직 끝 ---

                print(f"Epoch [{epoch+1}/{self.config.num_epochs}], Loss: {loss.item():.4f}")
                loss_history.append(loss.item())
                idx += 1
            # End of epoch, calculate average loss
            if len(loss_history) == 0:
                print("No loss recorded in this epoch. Skipping average loss calculation.")
                continue
            if len(loss_history) > 100:
                loss_history = loss_history[-100:]
            # Calculate average loss for the epoch
            avg_train_loss = sum(loss_history) / len(loss_history)
            print(f"Epoch [{epoch+1}/{self.config.num_epochs}], Average Training Loss: {avg_train_loss:.4f}")
            self.scheduler.step(avg_train_loss)
            self.scheduler.get_last_lr()

            if (epoch + 1) % self.config.save_interval == 0:
                model_path = os.path.join(self.config.model_dir, f"model_epoch_{epoch+1}.pth")
                torch.save(self.model.state_dict(), model_path)
                print(f"Model saved to {model_path}")
        
        total_loss = sum(loss_history) / len(loss_history)
        print(f"Training complete. Average Loss: {total_loss:.4f}")

            

    def evaluate(self):
        """
        Evaluate the model on the validation set.
        """
        self.model.eval()
        total_loss = 0.0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for batch in self.train_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        print(f"Validation Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    config = Config()
    trainer = Trainer(config)
    trainer.train()