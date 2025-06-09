clear all; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is a wrapper for automatic segmentation
%   of multiple real-time MRI (rtMRI) videos, 
%   using grid lines.
% The parameters from the rtMRI videos are extracted
%   based on a single grid line system which is
%   constructed based on manually selected grid center
%   points, using wrapper_grid.m
% If you have any question, please email to
%   jangwon@usc.edu
% Dec 18 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


addpath('../code_fn');
addpath('../config');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PARAMETERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options;
paths;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE PARAMETERIZATION (BATCH)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load the grid line
load(fullfile(path.output_data_dir,'grid.mat'));

% load all files of rtMRI video (in mat format)
input_file_list1 = dir(fullfile(path.mri_data_dir, '*.mat'));
input_file_list = input_file_list1(~startsWith({input_file_list1.name}, '._'));
num_files = length(input_file_list);

% start batch processing for each file
for iFile = 1:num_files

    fbname = input_file_list(iFile).name(1:end-4); % basename of file
    disp(['Start segmentation of ' fbname '...' num2str(iFile) '/' num2str(num_files)]);

    % load the first mat file
    file_path = fullfile(path.mri_data_dir, [fbname '.mat']);
    data_mri_orig = f_load_data_from_mat(file_path);

    % find frame indeces of speech region only
    spch_frames = f_get_spch_frame_idx(data_mri_orig,fbname,path,opt);
    data_mri_spch = data_mri_orig(spch_frames,:);

    % MRI image denoising and intensity correction
    disp('Start image enhancement...');
    tic;
    data_mri = f_img_enh(data_mri_spch, opt);
    toc;

    % define bin points of each gridline and compute the pixel intensity for each bin
    data_mri2 = reshape(data_mri,[],opt.img.img_size(1),opt.img.img_size(2));
    grid = f_comp_int(data_mri2, grid, opt);

    % tracking of the lips and the larynx
    disp('Start tracking the lips and the larynx...');
    track_out = [];
    % (1-1) Initial (top of larynx) vocal airway boundary detection
    track_out = f_track_larynx(data_mri2, grid, track_out, opt);

    % (1-2) estimate the location of the final grid of upper airway (front-most grid of lips)
    track_out = f_track_lip(data_mri2, grid, track_out, opt);

    % plot and save result video
    fpath = fullfile(path.output_data_dir, [fbname '_lip_larynx']);
    f_plot_lip_larynx(data_mri_spch, grid, track_out, opt, fpath);

    % obtain the optimal airway-path
    disp('Start estimating airway-path lines...');
    opt_path = f_airway_path(data_mri2, grid, track_out, opt);

    % unsupersived smoothing of the airway path
    disp('Start smoothing of the estimated airway-path...');
    opt_path_sm = f_sm_airway_path(grid, opt_path, track_out, opt);
    
    % plot and save result video
    fpath = fullfile(path.output_data_dir, [fbname '_air_path']);
    f_plot_airway_path_sm(data_mri_spch, opt_path_sm, opt_path, grid, track_out, opt, fpath);

    % tissue-airway boundary bin detection
    disp('Start detection of the tissue-airway boundaries...');
    bnd_out = [];
    bnd_out = f_opt_bnd(data_mri2, grid, opt_path_sm, track_out, bnd_out, opt);

    % Smoothing of estimated boundaries
    % smoothing by rloess until the lip landmark point
    %   & constrain to the palate surface
    % smoothing by robust fit (rlowess) 
    %  - use fliplr function for handling the error at the final grid in the upper airway
    % options.pal 1: min for palate constraint (recommended for the inner boundary)
    %             2: mean for palate constraint (recommended for the outer boundary)
    %             3: original palate points without constraint
    % options.phar 1: median filtering, DCT (recommended for the outer boundary)
    %              2: DCT (recommended for the inner boundary)
    disp('Start smoothing of estimated boundaries...');
    % inner boundary
    opt.tmp.pal = 1;
    opt.tmp.phar = 2;
    [bnd_out.bin_lb_sm, bnd_out.bin_lb_i] = f_sm_bin(bnd_out.bin_lb, grid, opt);
    % outer boundary
    opt.tmp.pal = 2;
    opt.tmp.phar = 1;
    [bnd_out.bin_rt_sm, bnd_out.bin_rt_i] = f_sm_bin(bnd_out.bin_rt, grid, opt);

    % plot and save result video
    fpath = fullfile(path.output_data_dir, [fbname '_bnd']);
    f_plot_bnd(data_mri_spch, grid, track_out, bnd_out, opt, fpath); 

    % compute distance function for each grid
    disp('Start computing distance function...');
    pts_lb_sm = f_bin2pts(bnd_out.bin_lb_sm,grid,track_out.idx_lips,track_out.idx_larynx);
    pts_rt_sm = f_bin2pts(bnd_out.bin_rt_sm,grid,track_out.idx_lips,track_out.idx_larynx);
    [vt_d,vt_dL] = f_comp_dist(bnd_out.bin_lb_sm, bnd_out.bin_rt_sm, pts_lb_sm, pts_rt_sm);

    % plot MRI image with boundaries and distance function
    fpath = fullfile(path.output_data_dir, [fbname '_vtd']);
    f_plot_vtd(data_mri_spch, vt_d, vt_dL, pts_lb_sm, pts_rt_sm, grid, fpath, opt);

    %% clean the tissue-airway boundary based on vt_dL
    [bnd_out.bin_lb_c, bnd_out.bin_rt_c] = f_clean_bnd(bnd_out.bin_lb_sm, bnd_out.bin_rt_sm, vt_dL);

    %% save grid, tissue boundary variable, vocal tract initial and final boundaries variable
    disp('Save estimated parameters...');
    path.output_data_dir
    fname1 = fullfile(path.output_data_dir, [fbname '_out.mat']);
    fname1
    fname = strrep(fname1,'._','');
   
    save(fname,'spch_frames','data_mri','track_out','opt_path','opt_path_sm','bnd_out','vt_dL','opt','grid');
    attributes = fileattrib(fname);
    if attributes == 1
        disp('Write permission is available');
    else
        dist('Write premission denied');
    end
    disp(['Done for ' fbname '...']);

end   % iFile

