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
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from utils import get_spectrogram, estimate_spectrogram_spl, AudioDataset, pack_collate_fn
import utils

class CNNAutoEncoder(nn.Module):
  def __init__(self):
    super(CNNAutoEncoder, self).__init__()
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0,return_indices=True)  # Add pooling layer
    self.layer1 = nn.Sequential(
      nn.Conv2d(1,20,6,stride=2,padding=2),
      nn.BatchNorm2d(20),
      nn.ReLU()
    )
    self.layer2 = nn.Sequential(
      nn.Conv2d(20,40,6,stride=6,padding=1),
      nn.BatchNorm2d(40),
      nn.ReLU(),   
      )
    self.pool2 = nn.Sequential(
      nn.Conv2d(40,10,3,stride=1,padding=1),
      nn.BatchNorm2d(10),
      nn.ReLU()
    )
    self.layer3 = nn.Sequential(
      nn.Conv2d(10,1,3,stride=1,padding=1),
      nn.Sigmoid()
    )
    self.layer4 = nn.Sequential(
      nn.Conv2d(1,10,3,stride=1,padding=1),
      nn.BatchNorm2d(10),
      nn.ReLU()
    )
    self.unpool2 = nn.Sequential(
      nn.ConvTranspose2d(10,40,3,stride=1,padding=1),
      nn.BatchNorm2d(40),
      nn.ReLU()
      )
    self.layer5 = nn.Sequential(
      nn.ConvTranspose2d(40,20,6,stride=6,padding=1),
      nn.BatchNorm2d(20),
      nn.ReLU()
      )
    self.layer6 = nn.Sequential(
      nn.ConvTranspose2d(20,1,6,stride=2,padding=2),
      nn.BatchNorm2d(1),
      nn.ReLU()
    )
    self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)  # Unpooling layer

  def forward(self, x):
    x, indices1 = self.pool1(x)  # Apply pooling and store indices
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.pool2(x)  # Apply pooling and store indices
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.unpool2(x)  # Unpool using stored indices
    x = self.layer5(x)
    x = self.layer6(x)
    x = self.unpool1(x, indices1)  # Unpool using stored indices
    return x


class Encoder(CNNAutoEncoder):
    def __init__(self):
        super(Encoder, self).__init__()
        

    def forward(self, x):
        x, indices1 = self.pool1(x)  # Apply pooling and store indices
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool2(x)  # Apply pooling and store indices
        x = self.layer3(x)
        return x  # Return the encoded representation and indices for unpooling
    
class Decoder(CNNAutoEncoder):
    def __init__(self):
        super(Decoder, self).__init__()
    
    def forward(self, x):
        x = self.unpool2(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x


class ImageDataset(AudioDataset):
    def __init__(self, root_dir, npz_dir, sample_rate=16000):
        super().__init__(root_dir, npz_dir, sample_rate)
        self.load_data_length()  # Load data when initializing the dataset
    
    def load_data_length(self):
        path_list = self.npz_path_list
        if not path_list:
            raise ValueError("No .npz files found in the specified directory.")
        self.data_length = []
       
        for npz_path in path_list:
            df = np.load(npz_path)
            if 'vt_image' in df:
                data = df['vt_image']
                if isinstance(data, np.ndarray):
                    self.data_length.append(data.shape[0])
                else:
                    raise ValueError(f"Data in {npz_path} is not a numpy array.")
            else:
                raise KeyError(f"No 'vt_image' key found in {npz_path}.")
        self.data_length = np.cumsum(self.data_length)
        
        return
    
    def __len__(self):
        return self.data_length[-1]


    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of range for the dataset.")
        for i, length in enumerate(self.data_length):
            if idx < length:
                if i == 0:
                    idx = idx
                else:
                    idx = idx - self.data_length[i-1]
                break
        npz_path = self.npz_path_list[i]
        df = np.load(npz_path)
        if 'vt_image' not in df:
            raise KeyError(f"No 'vt_image' key found in {npz_path}.")
        data = df['vt_image']
        if not isinstance(data, np.ndarray):
            raise ValueError(f"Data in {npz_path} is not a numpy array.")
        if idx >= data.shape[0]:
            raise IndexError(f"Index {idx} is out of bounds for data with shape {data.shape}.")
        data = data[idx]
        assert data.ndim == 2, f"Expected 2D array, got {data.ndim}D array."
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        
        return data, data


def encode_image_data():
    """
    Encode the image data using the trained encoder.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained encoder model
    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load(f'imageencoder/image_encoder_epoch{5}.pth'))
    encoder.eval()
    
    # Create the dataset and dataloader
    dataset = AudioDataset('Dataset', 'Dataset/npz', sample_rate=16000)
    base_dir = dataset.npz_dir.replace('npz','npz_encoded')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    for i, npz_path in enumerate(dataset.npz_path_list):
        print(f"Processing file {i+1}/{len(dataset.npz_path_list)}: {npz_path}")
        df = np.load(npz_path)
        if 'vt_image' not in df:
            print(f"No 'vt_image' key found in {npz_path}. Skipping.")
            continue
        data = df['vt_image']
        if not isinstance(data, np.ndarray):
            print(f"Data in {npz_path} is not a numpy array. Skipping.")
            continue
        if data.ndim != 3:
            print(f"Expected 3D array, got {data.ndim}D array in {npz_path}. Skipping.")
            continue
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
        data = data.to(device)
        data_list = []
        for j in range(data.shape[0]):
            with torch.no_grad():
                encoded_data = encoder(data[j].unsqueeze(0))  # Add batch dimension
                encoded_data = encoded_data.squeeze(0)
                if encoded_data.ndim == 3:
                    encoded_data = encoded_data.squeeze(0)
                encoded_data = encoded_data.reshape(encoded_data.shape[0],-1)  # Flatten the image to 2D
                data_list.append(encoded_data.cpu())
        
        encoded_data = torch.stack(data_list, dim=0)
        print(f"Encoded data shape for {npz_path}: {encoded_data.shape}")
        npz_dict = {
        'audio': df['audio'],  # Audio data
        'vt_image': encoded_data.cpu().numpy(),  # Vocal tract image
        'f0': df['f0'],  # F0 data
        'spl': df['spl'],  # SPL data
        }
        np.savez(os.path.join(base_dir,os.path.basename(npz_path)), **npz_dict)

    return

train = True  # Set to True for training, False for testing
if __name__ == "__main__" and train==True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = CNNAutoEncoder().to(device)
    
    dataset = ImageDataset('Dataset', 'Dataset/npz', sample_rate=16000)
    print(f"Dataset length: {len(dataset)}")
    print(len(dataset.npz_path_list))

    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    criterion = nn.MSELoss()
    
    # optimizer
    warmup_epoch = 2
    num_epochs = 20
    init_lr = 1e-1
    lr_lower_limit = 0

    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-3, betas=(0.9, 0.999), eps=1e-08)
    n_step_warmup = len(dataloader) * warmup_epoch
    total_iter = len(dataloader) * num_epochs
    iterations = 0
    
    encoder.train()
    for epoch in tqdm(range(num_epochs)):
        for images, _ in tqdm(dataloader,desc="epoch %d, iters" % (epoch + 1), leave=False):
            # lr cos schedule
            iterations += 1
            if iterations <= n_step_warmup:
                lr = init_lr * iterations / n_step_warmup
            else:
                lr = lr_lower_limit + 0.5 * (init_lr - lr_lower_limit) * (
                        1
                        + np.cos(
                            np.pi * (iterations - n_step_warmup) / (total_iter - n_step_warmup)
                        )
                    )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            images = images.to(device)
            optimizer.zero_grad()
            outputs = encoder(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
        
            print(f"\nEpoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}\n")
        print("Saving model at epoch %d" % (epoch + 1))
        if not os.path.exists('imageencoder'):
            os.makedirs('imageencoder')
        # Save the model state
        torch.save(encoder.state_dict(), f'imageencoder/image_encoder_epoch{epoch+1}.pth')    


if __name__ == "__main__" and train==False:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = CNNAutoEncoder().to(device)
    encoder.load_state_dict(torch.load(f'imageencoder/image_encoder_epoch{5}.pth'))
    encoder.eval()
    print("Encoder model loaded successfully.")
    dataset = ImageDataset('Dataset', 'Dataset/npz', sample_rate=16000)
    print(f"Dataset length: {len(dataset)}")
    print(len(dataset.npz_path_list))

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.MSELoss()
    dataiter = iter(dataloader)
    images, _ = next(dataiter)
    images = images.to(device)
    outputs = encoder(images)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(4):
        axes[0, i].imshow(images[i].cpu().squeeze().detach().numpy(), cmap='gray')
        axes[0, i].set_title('Original Image')
        axes[0, i].axis('off')
        axes[0, i].invert_xaxis()  # Invert x-axis for image display
        axes[0, i].invert_yaxis()  # Invert y-axis for image display

        axes[1, i].imshow(outputs[i].cpu().squeeze().detach().numpy(), cmap='gray')
        axes[1, i].set_title('Reconstructed Image')
        axes[1, i].axis('off')
        axes[1, i].invert_xaxis()  # Invert x-axis for image display
        axes[1, i].invert_yaxis()  # Invert y-axis for image display
    plt.suptitle('Original vs Reconstructed Images')
    plt.subplots_adjust(top=0.85)
    plt.tight_layout()
    plt.show()
    plt.savefig('./reconstructed_images.png')

    encode_image_data()
    # encode_image_data('Dataset/npz', 'Dataset/npz_encoded')
    # dataset = ImageDataset('Dataset', 'Dataset/npz_encoded', sample_rate=16000)
    # print(f"Dataset length: {len(dataset)}")
    # print(len(dataset.npz_path_list))
