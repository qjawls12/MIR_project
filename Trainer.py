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
                                pack_collate_fn, vt_loss, get_f0, f0_loss, spl_mse_loss


class Config:
    def __init__(self):
        """
        Configuration class to hold hyperparameters and paths.
        """
        self.gpu = 0
        self.calc_gpu = 0
        self.use_amp = False
        self.batch_size = 16
        self.learning_rate = 1e-3
        self.num_epochs = 50
        self.save_interval = 1
        self.model_dir = "models"
        self.model_name = "model.pth"
        self.epoch = 29
        self.load_model = True  # Set to True if you want to load a pre-trained model
        self.train_data_path = "Dataset"
        self.temperature = 10.0
        self.loss_dir = "loss_history"
        self.valid_dir = "valid_data"
        if self.load_model:
            self.start_epoch = self.epoch
        else:
            self.start_epoch = 0
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.loss_dir):
            os.makedirs(self.loss_dir)
        if not os.path.exists(self.valid_dir):
            os.makedirs(self.valid_dir)


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
        self.initial_losses = {'vt': None, 'f0': None, 'spl': None}
        if self.config.load_model:
            self.initial_losses['vt'] = torch.tensor(0.9218351244926453, dtype=torch.float32,
                                                     device=self.calc_device)
            self.initial_losses['f0'] = torch.tensor(0.3032028079032898, dtype=torch.float32,
                                                     device=self.calc_device)
            self.initial_losses['spl'] = torch.tensor(0.0931912288069725, dtype=torch.float32,
                                                      device=self.calc_device)

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
                self.model.load_state_dict(torch.load(model_path, weights_only=False, map_location=self.device))
                print(f"Model loaded from {model_path}")
            else:
                print(f"Model file {model_path} does not exist. Starting with a new model.")

    def loss_function(self, est, target, initial_losses:dict,temperature:float):
        """
        Custom loss function that computes the total loss with dynamic weighting.
        
        Args:
            outputs: Model outputs.
            targets: Ground truth targets.
            initial_losses: Initial losses for vt, f0, and spl.
        
        Returns:
            Total loss value.
        """
        vt_target = target[0]
        vt_target = vt_target.to(self.calc_device)  # Move to calc_device
        f0_target = target[1]
        f0_target = f0_target.to(self.calc_device)  # Move to calc_device
        spl_target = target[2]
        spl_target = spl_target.to(self.calc_device)  # Move to calc_device
        vt_loss_value = vt_loss(est[:,:,4:], vt_target)
        f0_loss_value = f0_loss(est[:,:,:3], f0_target)
        spl_loss_value = spl_mse_loss(est[:,:,3], spl_target) # 새로운 spl_mse_loss 사용
    
        current_losses = {
            'vt': vt_loss_value,
            'f0': f0_loss_value,
            'spl': spl_loss_value # 키 이름도 변경
        }

        # 첫 에포크 (또는 초기화가 필요한 시점)에서 초기 손실 값 저장
        if initial_losses['vt'] is None:
            for key in initial_losses:
                self.initial_losses[key] = current_losses[key].detach()

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
                # if torch.isnan(relative_loss).any() or torch.isinf(relative_loss).any():
                print(f"DEBUG: NaN/Inf in relative_loss for {key}. current: {current_losses[key].item()}, initial: {initial_losses[key].item()}")
                    # raise ValueError(f"NaN/Inf in relative_loss for {key}")

                exp_term = torch.exp(relative_loss / temperature)
            
                # 여기서 exp_term에 NaN/inf가 있는지 디버깅 출력 추가
                # if torch.isnan(exp_term).any() or torch.isinf(exp_term).any():
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
        spl_est = spl_est.to(self.calc_device)  # Move to calc_device
        spl_est = torch.log(spl_est + 1e-8)  # Logarithm for SPL estimation
        spl_est -= torch.max(spl_est, dim=-1, keepdim=True).values  # Normalize to zero mean
        spl_est = spl_est/(-torch.min(spl_est, dim=1, keepdim=True).values + eps)
        spl_tensor -= torch.max(spl_tensor, dim=-1, keepdim=True).values  # Normalize to zero mean
        spl_tensor = spl_tensor/(-torch.min(spl_tensor, dim=1, keepdim=True).values + eps)
        spl_est = spl_est.to(self.calc_device)  # Move to calc_device
        spl_tensor = spl_tensor.to(self.calc_device)  # Move to calc_device
        assert vt_tensor.shape == vt_est.shape, "Batch size mismatch between target and estimate for vt"
        assert f0_tensor.shape == f0_est.shape, "Batch size mismatch between target and estimate for f0"
        assert spl_tensor.shape == spl_est.shape, "Batch size mismatch between target and estimate for spl"
        vt_acc_list = []
        f0_acc_list = []
        spl_acc_list = []
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
        vt_acc = 100*sum(vt_acc_list) / len(vt_acc_list)
        f0_acc = 100*sum(f0_acc_list) / len(f0_acc_list)
        spl_acc = 100*sum(spl_acc_list) / len(spl_acc_list)
        
        total_acc = (vt_acc + f0_acc + spl_acc) / 3.

        return vt_acc, f0_acc, spl_acc, total_acc

            

            



    def train(self):
        """
        Train the model.
        """
        torch.cuda.empty_cache()
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2)


        loss_history = []
        for epoch in tqdm(range(self.config.start_epoch,self.config.num_epochs), desc="Training Epochs", leave=False):
            if self.config.load_model and epoch == self.config.start_epoch:
                print(f"Starting epoch {epoch+1}/{self.config.num_epochs}")
                losses = pd.read_csv(os.path.join(self.config.loss_dir, f"loss_history_epoch_{epoch}.csv"))['Loss'].tolist()
                loss_history.extend(losses)
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}", leave=False):
                idx = 0
                inputs, targets = batch
                inputs = inputs.to(self.calc_device)
                vt_image, f0s, spls = targets
                # Move vt_image, f0s, spls to calc_device
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
                    loss = self.loss_function(outputs, targets_on_device, self.initial_losses, self.config.temperature)
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
            # Calculate average loss for the epoch
            avg_train_loss = sum(loss_history) / len(loss_history)
            print(f"Epoch [{epoch+1}/{self.config.num_epochs}], Average Training Loss: {avg_train_loss:.4f}")
            self.scheduler.step(avg_train_loss)
            self.scheduler.get_last_lr()
            plt.plot(loss_history, label='Training Loss')
            plt.xlabel('iteration')
            plt.ylabel('Loss')
            plt.title(f'Training Loss History - Epoch {epoch+1}')
            plt.legend()
            plt.savefig(os.path.join(self.config.loss_dir, f"loss_history_epoch_{epoch+1}.png"))
            plt.close()
            pd.DataFrame(loss_history, columns=['Loss']).to_csv(os.path.join(self.config.loss_dir, f"loss_history_epoch_{epoch+1}.csv"), index=False)

            if (epoch + 1) % self.config.save_interval == 0:
                model_path = os.path.join(self.config.model_dir, f"model_epoch_{epoch+1}.pth")
                torch.save(self.model.state_dict(), model_path)

                print(f"Model saved to {model_path}")
            # validate
            valid_loss, vt_acc, f0_acc, spl_acc, total_acc = self.validate()

            # Save validation results to CSV
            valid_loss = valid_loss.item().cpu() if isinstance(valid_loss, torch.Tensor) else valid_loss
            vt_acc = vt_acc.cpu().item() if isinstance(vt_acc, torch.Tensor) else vt_acc
            f0_acc = f0_acc.cpu().item() if isinstance(f0_acc, torch.Tensor) else f0_acc
            spl_acc = spl_acc.cpu().item() if isinstance(spl_acc, torch.Tensor) else spl_acc
            total_acc = total_acc.cpu().item() if isinstance(total_acc, torch.Tensor) else total_acc
            pd.DataFrame({
                'Epoch': [epoch + 1],
                'Validation Loss': [valid_loss],
                'VT Accuracy': [vt_acc],
                'F0 Accuracy': [f0_acc],
                'SPL Accuracy': [spl_acc],
                'Total Accuracy': [total_acc]
            }).to_csv(os.path.join(self.config.valid_dir, f"validation_results_epoch_{epoch+1}.csv"), index=False)

        
        total_loss = sum(loss_history) / len(loss_history)
        print(f"Training complete. Average Loss: {total_loss:.4f}")

        return

            

    def validate(self):
        """
        Evaluate the model on the validation set.
        """
        self.model.eval()
        
        
        with torch.no_grad():
            valid_loss = []
            vt_acc_list = []
            f0_acc_list = []
            spl_acc_list = []
            total_acc_list = []
            for batch in self.val_loader:
                inputs, targets = batch
                inputs, inputs.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_function(outputs, targets, self.initial_losses, self.config.temperature)
                valid_loss.append(loss.item())
                vt_acc, f0_acc, spl_acc, total_acc = self.accuracy_function(outputs, targets)
                vt_acc_list.append(vt_acc)
                f0_acc_list.append(f0_acc)
                spl_acc_list.append(spl_acc)
                total_acc_list.append(total_acc)

            
            # Calculate and print validation loss
        valid_loss = sum(valid_loss)
        vt_acc = sum(vt_acc_list) / len(vt_acc_list)
        f0_acc = sum(f0_acc_list) / len(f0_acc_list)
        spl_acc = sum(spl_acc_list) / len(spl_acc_list)
        total_acc = sum(total_acc_list) / len(total_acc_list)
        if len(self.val_loader) == 0:
            print("No validation data available. Skipping validation loss calculation.")
            return
        avg_loss = valid_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_loss:.4f}")
        print(f"Validation VT Accuracy: {vt_acc:.2f}%")
        print(f"Validation F0 Accuracy: {f0_acc:.2f}%")
        print(f"Validation SPL Accuracy: {spl_acc:.2f}%")
        print(f"Validation Total Accuracy: {total_acc:.2f}%")
        return avg_loss, vt_acc, f0_acc, spl_acc, total_acc


if __name__ == "__main__":
    config = Config()
    trainer = Trainer(config)
    trainer.validate()
    trainer.train()