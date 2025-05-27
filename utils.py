import os
import torch
import torchaudio
import librosa
import numpy as np
import scipy.io


# 데이터셋 다운로드
path = '/Volumes/One Touch/USC-TIMIT/MRI/Data/F1/tracks'
mat = scipy.io.loadmat(path + '/usctimit_mri_f1_001_005_track.mat')
print('Dataset loaded successfully!')
print(mat['trackdata'].dtype)  # 데이터의 형태 출력
#print(type(mat['trackdata']))           # <class 'numpy.ndarray
#print(type(mat['trackdata'][0,0]))    # 구조체라면 <class 'numpy.void'> 또는 object
first_item = mat['trackdata'][0, 0]
#print(type(first_item))
#print(first_item.shape)   # 내부 데이터 형태
#print(first_item.dtype)   # 자료형
data = mat['trackdata'][0, 0]  # 첫 번째 항목의 데이터 추출
# print(data['contours'])
# print(data['frameNo'])
# print(first_item[:5])
print(data['contours']['segment'].dtype)  # 첫 번째 프레임의 첫 번째 객체의 contour 좌표
#data['contours'][0, 0][0][0]['i']  # 군집 또는 클래스 인덱스
#data['contours'][0, 0][0][0]['mu']
# 데이터셋 다운로드 함수
