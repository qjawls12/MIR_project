import os
import subprocess
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
from utils import AudioDataset

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
        self.load_state_dict(torch.load(f'imageencoder/image_encoder_epoch{20}.pth', weights_only=True, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
        

    def k_means_clustering(self):
        """
        Perform k-means clustering on the encoded data.
        """
        # Load the encoded data
        encoded_data = np.load('Dataset/npz_encoded/encoded_data.npz')
        vt_image = encoded_data['vt_image']
        
        # Perform k-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=128, random_state=36)
        kmeans.fit(vt_image.reshape(vt_image.shape[0], -1))
        labels = kmeans.labels_
        print(f"K-means clustering completed with {kmeans.n_clusters} clusters.")



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
        self.load_state_dict(torch.load(f'imageencoder/image_encoder_epoch{20}.pth', weights_only=True, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
        

    def numpy_array_to_video_ffmpeg(
        self,
        numpy_array: np.ndarray,
        output_filename: str,
        fps: int = 23.18,
        codec: str = "libx264", # 비디오 코덱 (H.264)
        pixel_format_out: str = "yuv420p", # 출력 픽셀 포맷 (대부분의 플레이어에서 호환)
        pixel_format_in: str = "rgb24" # ffmpeg에 전달할 입력 픽셀 포맷
        ):
        """
        (number_of_frame, height, width, depth) 크기의 NumPy 배열을 동영상 파일로 변환합니다.

        Args:
        numpy_array (np.ndarray): 입력 NumPy 배열.
                                  형태: (frames, height, width, channels)
                                  channels: 3 (RGB) 또는 4 (RGBA)
                                  데이터 타입: uint8 (0-255)이어야 합니다.
        output_filename (str): 출력될 동영상 파일의 이름 (예: "output.mp4").
        fps (int): 초당 프레임 수.
        codec (str): 사용할 비디오 코덱. 기본값은 "libx264" (MP4).
        pixel_format_out (str): 출력 동영상의 픽셀 포맷. 기본값은 "yuv420p" (넓은 호환성).
        pixel_format_in (str): ffmpeg가 NumPy 배열에서 받을 것으로 예상하는 입력 픽셀 포맷.
                                기본값은 "rgb24" (RGB 8비트).
        """

        if numpy_array.dtype != np.uint8:
            print(f"경고: NumPy 배열의 데이터 타입이 {numpy_array.dtype}입니다. uint8로 변환합니다.")
            # NumPy 배열의 값을 0-255 범위로 스케일링하여 uint8로 변환
            # float 타입이라면 0-1 범위로 정규화된 것을 0-255로 변환
        if np.issubdtype(numpy_array.dtype, np.floating):
            numpy_array = (numpy_array.clip(0, 1) * 255).astype(np.uint8)
        else: # 다른 타입이라면 단순히 uint8로 변환 (값 범위는 유지된다고 가정)
            numpy_array = numpy_array.astype(np.uint8)


        num_frames, height, width, channels = numpy_array.shape

        if channels == 3:
            pixel_format_in = "rgb24"
        elif channels == 4:
            pixel_format_in = "rgba"
        else:
            raise ValueError("채널 수는 3 (RGB) 또는 4 (RGBA)여야 합니다.")

        # ffmpeg 명령 구성
        command = [
            'ffmpeg',
            '-y',  # 출력 파일이 존재하면 덮어쓰기
            '-f', 'rawvideo',  # 입력 포맷: raw video
            '-vcodec', 'rawvideo', # 입력 비디오 코덱: raw video
            '-s', f'{width}x{height}',  # 프레임 크기
            '-pix_fmt', pixel_format_in,  # 입력 픽셀 포맷 (예: rgb24, rgba)
            '-r', str(fps),  # 입력 프레임 레이트
            '-i', '-',  # 입력은 stdin (표준 입력)에서 받음
            '-c:v', codec,  # 출력 비디오 코덱
            '-pix_fmt', pixel_format_out,  # 출력 픽셀 포맷 (예: yuv420p, argb)
            '-loglevel', 'warning', # 경고 레벨만 표시
            output_filename
            ]

        print(f"ffmpeg 명령: {' '.join(command)}")

        # subprocess를 사용하여 ffmpeg 프로세스 시작
        # stdin=subprocess.PIPE로 설정하여 NumPy 데이터를 파이프로 전달
        # stdout=subprocess.PIPE, stderr=subprocess.PIPE로 설정하여 ffmpeg의 출력을 캡처 (디버깅용)
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # NumPy 배열 데이터를 ffmpeg 프로세스의 stdin으로 전달
        # 각 프레임은 (height * width * channels) 크기의 바이트열이어야 합니다.
        # .tobytes()를 사용하여 NumPy 배열을 바이트열로 직렬화합니다.
        for i in range(num_frames):
            process.stdin.write(numpy_array[i].tobytes())

        # stdin을 닫고 ffmpeg 프로세스가 완료될 때까지 기다립니다.
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"ffmpeg 에러: {stderr.decode('utf-8')}")
            raise RuntimeError("ffmpeg 변환 실패")
        else:
            print(f"'{output_filename}'으로 동영상 변환 성공.")
    
        return
    
    def forward(self, x):
        x = self.layer4(x)
        x = self.unpool2(x)
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
    encoder.load_state_dict(torch.load(f'imageencoder/image_encoder_epoch{20}.pth'))
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





train = False  # Set to True for training, False for testing
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
    encoder.load_state_dict(torch.load(f'imageencoder/image_encoder_epoch{20}.pth',weights_only=True))
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


    # encode_image_data()
    # encode_image_data('Dataset/npz', 'Dataset/npz_encoded')
    # dataset = ImageDataset('Dataset', 'Dataset/npz_encoded', sample_rate=16000)
    # print(f"Dataset length: {len(dataset)}")
    # print(len(dataset.npz_path_list)
