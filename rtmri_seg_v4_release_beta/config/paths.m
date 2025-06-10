%% Paths
% - Assume that the rtMRI videos are in mat data format of N x M matrix
%     N x M; N is the number of frame, M is the number of pixels
% - directory path for the silence labels: [] if no silence label is used.
%     The silence label should be in the following format: starting time (tab) ending time
%     See examples in ../rtMRIdata/lab_sil
path.mri_data_dir = '/Volumes/One_Touch/rtMRI_Dataset/VOCOLAB/Singer10/mat';
path.output_data_dir = '/Volumes/One_Touch/rtMRI_Dataset/VOCOLAB/Singer10/out/seg';
path.morph_data_dir = '/Volumes/One_Touch/rtMRI_Dataset/VOCOLAB/Singer10/out/morph';
path.sil_lab = '/Volumes/One_Touch/rtMRI_Dataset/VOCOLAB/Singer10/out/lab_sil';
path.avi_data_dir = '/Volumes/One_Touch/rtMRI_Dataset/VOCOLAB/avi/Singer10';


% path.mri_data_dir = '/Volumes/One_Touch/MRI/Data/F2/mat';
% path.output_data_dir = '/Volumes/One_Touch/MRI/Data/F2/out/seg';
% path.morph_data_dir = '/Volumes/One_Touch/MRI/Data/F2/out/morph';
% path.sil_lab = '/Volumes/One_Touch/MRI/Data/F2/out/lab_sil';
% path.avi_data_dir = '/Volumes/One_Touch/MRI/Data/F2/avi';
