clear all; clc;

% This code flip MR images upside down for all mat files in a certain directlry

% If you have any question, please email to jangwon@usc.edu
% Jangwon Kim
% May 2nd 2014


dir_data = '../rtMRIdata/mat_init';
dir_out = '../rtMRIdata/mat';

mkdir(dir_out);

list = dir(fullfile(dir_data, '*.mat'));

for iFile = 1:length(list)
  iFile

  M = f_load_data_from_mat(fullfile(dir_data, list(iFile).name));
  mri_data = zeros(size(M));
  for iFrame = 1:size(M,1);
    mri_data(iFrame,:) = reshape(flipud(reshape(M(iFrame,:),68,68)),1,[]);
  end

  %h=figure('Position',[100 100 500 250]); 
  %subplot(1,2,1); imagesc(reshape(M(1,:),68,68)); title('before');
  %subplot(1,2,2); imagesc(reshape(mri_data(1,:),68,68)); title('after'); keyboard; close(h);

  save(fullfile(dir_out, list(iFile).name),'mri_data');
end
