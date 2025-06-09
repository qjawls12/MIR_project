clear all; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is a wrapper for automatic parameterization
%   for morphological characteristics
%   in multiple real-time MRI (rtMRI) videos, 
%   using grid lines.
% The parameters from the rtMRI videos are extracted
%   based on tissue-ariway boundaries
%   generated using wrapper_seg_batch.m
% If you have any question, please email to
%   jangwon@usc.edu
% Feb 14 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../code_fn');
addpath('../config');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PARAMETERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
paths;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MORPHOLOGICAL PARAMETER EXTRACTION (BATCH)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

system(['mkdir -p ' path.morph_data_dir]);

% load the grid line
input_file_list1 = dir(fullfile(path.output_data_dir, ['*_out.mat']));
input_file_list = input_file_list1(~startsWith({input_file_list1.name}, '._'));
num_files = length(input_file_list);

for iFile = 1:num_files

    fbname1 = input_file_list(iFile).name(1:end-8); % basename of file
    fbname = strrep(fbname1,'._','');
    fullpath = fullfile(path.output_data_dir, [fbname '_out.mat']);
    fullpath

    % load segmentation output
    load(fullfile(path.output_data_dir,[fbname '_out.mat']));

    % determine the center lines in the upper airway
    pts_lb_c = f_bin2pts(bnd_out.bin_lb_c,grid,track_out.idx_lips,track_out.idx_larynx);
    pts_rt_c = f_bin2pts(bnd_out.bin_rt_c,grid,track_out.idx_lips,track_out.idx_larynx);
    [pts_c, vtl] = f_det_center(pts_lb_c, pts_rt_c, opt);

    %% save speech frames, center lines,the vocal tract length
    disp('Save estimated parameters...');
    fname = fullfile(path.morph_data_dir, [fbname '_morph.mat']);
    save(fname,'spch_frames','pts_c','vtl','pts_lb_c','pts_rt_c');
    disp(['Done for ' fbname '...']);

end % iFile
