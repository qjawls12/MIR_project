% clear all; clc;

% convert format of each file from avi to mat

% If you have any question, please email to jangwon@usc.edu
% Jangwon Kim
% May 2nd 2014

% config;
% path;

input_dir = path.avi_data_dir;		% directory of input avi data
out_dir = path.mri_data_dir;	% directory of output mat data

list_file1 = dir(fullfile(input_dir, '*.avi'));
list_file = list_file1(~startsWith({list_file1.name}, '._'));

num_file = length(list_file);

for iFile = 1:num_file
  fpath = fullfile(input_dir, list_file(iFile).name);
  mri_data = Avi2MovieMat_jw(fpath);

  fpath = fullfile(out_dir, [list_file(iFile).name(1:end-4) '.mat']);
  save(fpath,'mri_data');
end
