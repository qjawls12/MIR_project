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
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils import get_spectrogram, estimate_fundamental_frequency

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    # Forward pass for the Transformer block
    # x: input tensor of shape (batch_size, seq_length, embed_dim)
    # mask: boolean mask of shape (batch_size, seq_length) indicating padded positions
    # Returns: output tensor of shape (batch_size, seq_length, embed_dim)
    def forward(self, x, mask):
        x = self.layernorm1(x)  # Layer normalization before attention
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)
        if torch.isnan(attn_output).any():
            print(f"DEBUG: NaN/Inf in attn_output. Min: {attn_output.min().item()}, Max: {attn_output.max().item()}")

        x = x + self.dropout(attn_output)
        if torch.isnan(x).any():
            print(f"DEBUG: NaN/Inf after layernorm1. Min: {x.min().item()}, Max: {x.max().item()}")
        
        x = self.layernorm2(x)  # Layer normalization after attention
        ffn_output = self.ffn(x)
        if torch.isnan(ffn_output).any():
            print(f"DEBUG: NaN/Inf in ffn_output. Min: {ffn_output.min().item()}, Max: {ffn_output.max().item()}")

        x = x + self.dropout(ffn_output)
        if torch.isnan(x).any():
            print(f"DEBUG: NaN/Inf after layernorm2. Min: {x.min().item()}, Max: {x.max().item()}")
        return x
    # attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)
    # x = self.layernorm1(x + self.dropout(attn_output))
    # ffn_output = self.ffn(x)
    # x = self.layernorm2(x + self.dropout(ffn_output))

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.layernorm(x)



class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, dropout=0.1):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Initial hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Initial cell state
        out, _ = self.lstm(x, (h_0, c_0))
        return out


class Model(nn.Module):
    """input : spectrogram
       output: class probabilities and avtivation parameters"""
    def __init__(self, n_fft = 1382, hidden_dim = 128, output_dim = 40, device='cuda'):
        super(Model, self).__init__()
        self.device = device
        self.n_fft = n_fft
        self.spectrogram_dim = n_fft // 2 + 1  # Assuming n_fft is the FFT size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.preprocess = self.make_batch_length_mask
        self.mel_filterbank = torchaudio.functional.melscale_fbanks(n_freqs=self.spectrogram_dim,
                                                                    n_mels=hidden_dim,sample_rate=16000,
                                                                    f_min=0, f_max=8000,
                                                                    norm='slaney',mel_scale='htk').to(device=self.device, dtype=torch.float32)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80).to(device=self.device, dtype=torch.float32)
        self.layernorm = nn.LayerNorm(self.hidden_dim)
        self.transformer_encoder = TransformerEncoder(num_layers=4, embed_dim=hidden_dim, num_heads=32, ff_dim=hidden_dim * 2, dropout=0.01)
        self.lstm_block = LSTMBlock(hidden_size=hidden_dim, input_size=hidden_dim, num_layers=3)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    
    # audio to spectrogram conversion
    def audio_preprocess(self, audio_packed):
        audio, lengths = pad_packed_sequence(audio_packed, batch_first=True)
        # Convert audio to spectrogram
        spectrogram = []
        f0 = []
        for i in range(audio.size(0)):
            audio_i = audio[i].cpu().numpy()
            spec = get_spectrogram(audio_i)
            f0_i = estimate_fundamental_frequency(audio_i)
            f0_i = torch.tensor(f0_i, dtype=torch.float32)
            spectrogram.append(spec) #torch.tensor(spec).to(dtype=torch.float32))
            f0.append(f0_i)
        spectrogram = torch.stack(spectrogram, dim=0)
        f0 = torch.stack(f0, dim=0)
        
        return spectrogram, f0, lengths
   
    # Create a sequential length mask for the transformer block
    def make_batch_length_mask(self, audio_packed):
        # Create a mask for the transformer block
        specs, f0, lengths = self.audio_preprocess(audio_packed)
        max_length = specs.size(1)
        mask = torch.zeros(specs.size(0), specs.size(1), dtype=torch.bool, device=specs.device)
        for i, length in enumerate(lengths):
            if length < max_length:
                mask[i, length:] = True
            # mask[f0 == 0.0] = True # Set silent values to 0
        return specs.to(self.device), mask.to(self.device)
        

    # Forward pass
    # x_packed: PackedSequence of audio data
    # x: spectrogram (float32), mask: bool mask for padding
    # Returns: x: output probabilities (float32)
    def forward(self, x_packed): # 인자 이름을 x_packed로 변경
        # Step 0: Initial input check
        # Inputs (PackedSequence.data) dtype: torch.float32, device: cuda:0
        if torch.isnan(x_packed.data).any() or torch.isinf(x_packed.data).any():
            print(f"DEBUG: NaN/Inf in initial x_packed.data. Min: {x_packed.data.min().item()}, Max: {x_packed.data.max().item()}")

        x, mask = self.preprocess(x_packed) # x는 spectrogram (float32), mask는 bool
        # print(f"DEBUG: x shape: {x.shape}, mask shape: {mask.shape}")

        # Step 1: after self.preprocess (spectrogram, mask)
        # if torch.isnan(x).any() or torch.isinf(x).any():
        print(f"DEBUG: NaN/Inf in spectrogram after preprocess. Min: {x.min().item()}, Max: {x.max().item()}")
        if torch.isnan(mask).any(): # Mask는 bool이므로 NaN이 있을 수 없지만, 혹시 모를 타입 변환 오류 등 확인
             print(f"DEBUG: NaN in mask after preprocess.")
        if mask.all(dim=1).any():
            print("DEBUG: Warning! At least one sequence is entirely masked (all True in mask).")
        # print(f"DEBUG: Mask True ratio: {mask.sum().item() / mask.flatten().numel():.4f}")


        # Projection Layer: spectrogram_dim (float32) -> hidden_dim (float32)
        x = torch.matmul(x.float(), self.mel_filterbank)
        x = self.amplitude_to_db(x) # Convert to dB scale
        # dB 스케일로 변환된 spectrogram_db 사용
        min_val = torch.min(x, dim=1).values
        min_val = torch.min(min_val, dim=1).values.unsqueeze(1).unsqueeze(1)
        max_val = torch.max(x, dim=1).values  # (batch_size, hidden_dim)
        max_val = torch.max(max_val, dim=1).values.unsqueeze(1).unsqueeze(1)
        print(f"DEBUG: min_val shape: {min_val.shape}, max_val shape: {max_val.shape}")
        print(f"DEBUG: min_val: {min_val.min().item()}, max_val: {max_val.max().item()}")
        print(f"DEBUG: x shape before projection: {x.shape}, dtype: {x.dtype}")

        # min_val이 max_val과 같은 경우 (예: 모든 값이 동일한 경우) 0으로 나누기 방지
        x = (x - min_val) / (max_val - min_val+1e-6)
        print(f"{x.min.item()}, {x.max().item()}")
        # print(f"DEBUG: x shape after projection: {x.shape}, dtype: {x.dtype}")
        # x = self.sigmoid(x)  # Sigmoid activation
        # Step 2: after self.projection
        
        

        # Batch, time, hidden_dim
        # x = self.layernorm(x)  # Batch Normalization
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"DEBUG: NaN/Inf in x after projection. Min: {x.min().item()}, Max: {x.max().item()}")


        # AMP (autocast)가 여기서 float16으로 변환할 것입니다.
        # Transformer와 LSTM은 float16으로 연산합니다.
       
        if x.shape[1] > 128: # 시퀀스 길이에 따라 처리
            x_list = []
            for i in range(0, x.shape[1], 128):
                if i + 128 > x.shape[1]:
                    # 마지막 부분 처리
                    current_x = x[:, i:, :]
                    current_mask = mask[:, i:]
                else:
                    # 128 길이씩 슬라이딩
                    current_x = x[:, i:i+128, :]
                    current_mask = mask[:, i:i+128]
                # if current_mask.all(dim=1).any():
                #     x_list.append(current_x)  # 현재 슬라이스가 모두 마스크된 경우, 그냥 추가

                # if current_mask.sum().item() / current_mask.flatten().numel() > 0.8:
                #     x_list.append(current_x)  # 현재 슬라이스가 80% 이상 활성화된 경우, 그냥 추가

                # Step 3a: before transformer_encoder inside loop
                if torch.isnan(current_x).any() or torch.isinf(current_x).any():
                    print(f"DEBUG: NaN/Inf in current_x before transformer_encoder (loop). Min: {current_x.min().item()}, Max: {current_x.max().item()}")
                
                # Transformer Encoder Layer
                x_i = self.transformer_encoder(current_x, mask=current_mask)

                # Step 3b: after transformer_encoder inside loop
                if torch.isnan(x_i).any() or torch.isinf(x_i).any():
                    print(f"DEBUG: NaN/Inf in x_i after transformer_encoder (loop). Min: {x_i.min().item()}, Max: {x_i.max().item()}")
                    # 문제가 발생했다면 여기서 print될 것입니다.
                    # 다음 스텝으로 NaN이 전파되지 않도록 임시로 NaN을 0으로 채워볼 수 있습니다 (디버깅용)
                    # x_i = torch.nan_to_num(x_i, nan=0.0, posinf=0.0, neginf=0.0)
                x_list.append(x_i)
            x = torch.cat(x_list, dim=1)
            # print(f"DEBUG: x shape after concatenation: {x.shape}, dtype: {x.dtype}")

            # LSTM Block
            x = self.lstm_block(x)
            # Step 4a: after lstm_block inside loop
            if torch.isnan(x_i).any() or torch.isinf(x_i).any():
                print(f"DEBUG: NaN/Inf in x_i after lstm_block (loop). Min: {x_i.min().item()}, Max: {x_i.max().item()}")
                # x_i = torch.nan_to_num(x_i, nan=0.0, posinf=0.0, neginf=0.0)
            print

            # Final FC layer
            x = self.fc(x)
            # Step 5a: after fc inside loop
            if torch.isnan(x_i).any() or torch.isinf(x_i).any():
                print(f"DEBUG: NaN/Inf in x_i after fc (loop). Min: {x_i.min().item()}, Max: {x_i.max().item()}")
                # x_i = torch.nan_to_num(x_i, nan=0.0, posinf=0.0, neginf=0.0)

            # activation functions
            x[:,:,:3] = self.sigmoid(x[:,:,:3]) # Assuming the first 3 channels are for activation parameters
            x[:,:,3] = self.relu(x[:,:,3])  # Assuming the 4th channel is for real-valued output
            x[:,:,4:] = self.sigmoid(x[:,:,4:])  # Assuming the rest are activation parameters
            # Step 6a: after sigmoid inside loop
            if torch.isnan(x_i).any() or torch.isinf(x_i).any():
                print(f"DEBUG: NaN/Inf in x_i after sigmoid (loop). Min: {x_i.min().item()}, Max: {x_i.max().item()}")
                # x_i = torch.nan_to_num(x_i, nan=0.0, posinf=0.0, neginf=0.0)

                
            
        else:
            # Step 3c: before transformer_encoder (no loop)
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"DEBUG: NaN/Inf in x before transformer_encoder (no loop). Min: {x.min().item()}, Max: {x.max().item()}")

            x = self.transformer_encoder(x, mask=mask)
            # Step 3d: after transformer_encoder (no loop)
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"DEBUG: NaN/Inf in x after transformer_encoder (no loop). Min: {x.min().item()}, Max: {x.max().item()}")
                # x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            x = self.lstm_block(x)
            # Step 4b: after lstm_block (no loop)
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"DEBUG: NaN/Inf in x after lstm_block (no loop). Min: {x.min().item()}, Max: {x.max().item()}")
                # x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            x = self.fc(x)
            # Step 5b: after fc (no loop)
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"DEBUG: NaN/Inf in x after fc (no loop). Min: {x.min().item()}, Max: {x.max().item()}")
                # x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            x = self.sigmoid(x)
            # Step 6b: after sigmoid (no loop)
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"DEBUG: NaN/Inf in x after sigmoid (no loop). Min: {x.min().item()}, Max: {x.max().item()}")
                # x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Step 7: Final output check
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"DEBUG: NaN/Inf in final output x. Min: {x.min().item()}, Max: {x.max().item()}")
            # x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
          # Final fully connected layer
        return x
        # Final output
        
        # x, mask = self.preprocess(x)
        # x = self.projection(x)
        # if x.shape[1] > 256:
        #     x_list = []
        #     for i in range(0, x.shape[1], 256):
        #         if i + 256 < x.shape[1]:
        #             x_i = self.transformer_encoder(x[:, i:i+256, :], mask=mask[:, i:i+256])
        #             x_i = self.lstm_block(x_i)
        #             x_i = self.fc(x_i)
        #             x_i = self.sigmoid(x_i)
        #             x_list.append(x_i)
        #         else:
        #             x_i = self.transformer_encoder(x[:, i:, :], mask=mask[:, i:])
        #             x_i = self.lstm_block(x_i)
        #             x_i = self.fc(x_i)
        #             x_i = self.sigmoid(x_i)
        #             x_list.append(x_i)
        #     x = torch.cat(x_list, dim=1)
        # else:
        #     x = self.transformer_encoder(x, mask=mask)
        #     x = self.lstm_block(x)
        #     x = self.fc(x)
        #     x = self.sigmoid(x)

        # return x