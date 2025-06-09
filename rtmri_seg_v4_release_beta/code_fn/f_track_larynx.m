function track_out = f_track_larynx(data_mri, grid, track_out, opt)

% tracking the top of the larynx (the arytenoid cartilage) using the Viterbi algorithm
% The current version puts more weights on higher grid lines than lower grid lines
% If you have any question, please email to jangwon@usc.edu
% Jangwon Kim
% Oct 19th 2014

% Assign simpler variable name
search_length_larynx_mm = opt.lar.search_length_larynx_mm;
search_width_larynx_mm = opt.lar.search_width_larynx_mm;
px_size = opt.img.px_size;
gint = opt.grid.gint;
sigmoid_param = opt.airway.sigmoid_param_air_lip;
img_size = opt.img.img_size;
gwidth = opt.grid.gwidth;

num_frame = size(data_mri,1);
num_bin = size(grid.bin_pts,2);
num_grid = size(grid.bin_pts,1);

% larynx landmark point index in grid.center_pt
[~,grid_idx_larynx_landmark] = min(pdist2(grid.center_pt, grid.larynx));

search_length = round(search_length_larynx_mm/px_size); % length of each grid line (for one side)
search_width = round(search_width_larynx_mm/px_size);   % # grid lines in the up/down direction of the head (for one side)

search_bin_idx = round(num_bin/2) + (((-1)*search_length):1:search_length); % number of bins of search
search_grid_idx = grid_idx_larynx_landmark + (((-1)*search_width):1:search_width);
tmp_idx = find(search_grid_idx > num_grid);
search_grid_idx(1:tmp_idx) = [];

% compute derivative of mean intensity in the search region of each grid line
all_ave_int = zeros(num_frame, length(search_grid_idx));
all_ave_int_diff = zeros(num_frame, (length(search_grid_idx)-1));
for which_frame = 1:num_frame
  cur_frame_vt_data = reshape(grid.line_intensity(which_frame, search_grid_idx, search_bin_idx), length(search_grid_idx), length(search_bin_idx));
  ave_int = mean(cur_frame_vt_data, 2);
  all_ave_int(which_frame,:) = ave_int;
  all_ave_int_diff(which_frame,:) = (-1) * (ave_int(2:end) - ave_int(1:end-1)); % smallest (-) intensity derivative grid will be tracked.
end

% more weighting on the higher grid lines than lower grid lines
min_w = 0.5; % You may want to change this parameters depending on your weighting strategy.
max_w = 1;   % You may want to change this parameters depending on your weighting strategy.
w = linspace(max_w,min_w,(length(search_grid_idx)-1));
all_ave_int_diff = all_ave_int_diff .* repmat(w,num_frame,1);

% optimal path
num_bin = length(search_grid_idx)-1; % each search grid is a bin for this problem
% construct transition matrix (between bins of adjacent grid lines)
% transmat: the Euclidean distance between adjacent pixel
% construct transition matrix (between bins of adjacent grid lines)
tmp_transmat = zeros(num_bin,num_bin);
for which_pre_bin = 1:num_bin
    for which_pos_bin = 1:num_bin
        % the Euclidean distance between bins of adjacent grid lines
        tmp_transmat(which_pre_bin,which_pos_bin) = sqrt(abs(which_pre_bin - which_pos_bin)^2 + (gint^2));
    end
end

% normalize value to be in [0 1]
tmp_transmat_n = (tmp_transmat - min(min(tmp_transmat))) ./ (max(max(tmp_transmat)) - min(min(tmp_transmat)));

% sigmoid function warping of the transition matrix
transmat_w = f_sigmoid_warping(tmp_transmat_n,sigmoid_param);

% normalize value to be in [0 1]
transmat = cell(num_frame-1,1);
for which_frame = 1:(num_frame-1)
    transmat{which_frame} = (transmat_w - min(min(transmat_w))) ./ (max(max(transmat_w)) - min(min(transmat_w)));
end

% PRIOR
prior = all_ave_int_diff(1,:); % the value of the first frame

% OBJECT LIKELIHOOD
tmp_obslik = all_ave_int_diff; % derivatives of average pixel intensity
obslik = (tmp_obslik - min(min(tmp_obslik))) ./ (max(max(tmp_obslik)) - min(min(tmp_obslik)));

% viterbi decoding
opt_path = f_viterbi_path_min(prior,obslik,transmat);

% smoothing
tmp = search_grid_idx(opt_path) - 1; % choose the grid line right above the boundary
track_out.idx_larynx = tmp;
pos_larynx_x = smooth(grid.center_pt(tmp,1),3,'moving');
pos_larynx_y = smooth(grid.center_pt(tmp,2),3,'moving');
track_out.pos_larynx = [pos_larynx_x pos_larynx_y];

%% plot and check
%h=figure; 
%for which_frame = 1:num_frame
%  imagesc(reshape(data_mri(which_frame,:,:),img_size(1),img_size(2))); colormap(gray); hold on;
%  plot(track_out.pos_larynx(which_frame,1), track_out.pos_larynx(which_frame,2),'r.','MarkerSize',30);
%  pause((1/opt.img.fr_image)); hold off;
%end; close(h);

