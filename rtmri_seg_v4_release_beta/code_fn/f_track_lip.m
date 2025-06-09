function track_out = f_track_lip(data_mri2, grid, track_out, opt);

% tracking the front-most edge of the lips in the grid lines
% If you have any question, please email to jangwon@usc.edu
% Jangwon Kim
% Oct 20 2014

% Assign simpler variable name
search_length_lips_mm = opt.lip.search_length_lips_mm;
search_width_lips_mm = opt.lip.search_width_lips_mm;
px_size = opt.img.px_size;
gint = opt.grid.gint;
img_size = opt.img.img_size;
gwidth = opt.grid.gwidth;

num_frame = size(data_mri2,1);
num_bin = size(grid.bin_pts,2);
num_grid = size(grid.bin_pts,1);

% lips landmark point index in grid.center_pt
[~,grid_idx_lips_landmark] = min(pdist2(grid.center_pt, grid.mlab));

search_length = round(search_length_lips_mm/px_size);	% # grid lines for one direction (upward, downward)
search_width_up = round(search_width_lips_mm(1)/px_size);	% # bins for search (up)
search_width_down = round(search_width_lips_mm(2)/px_size);     % # bins for search (down)

search_bin_idx = round(num_bin/2) + (((-1)*search_width_down):1:search_width_up); % number of bins of search
search_grid_idx = grid_idx_lips_landmark + (((-1)*search_length):1:0);
tmp_idx = find(search_grid_idx < 1);
search_grid_idx(:,tmp_idx) = [];

% compute derivative of maximum intensity in the search region of each grid line
all_ave_int = zeros(num_frame, length(search_grid_idx));
all_ave_int_diff = zeros(num_frame, (length(search_grid_idx)-1));
for which_frame = 1:num_frame
  cur_frame_vt_data = reshape(grid.line_intensity(which_frame, search_grid_idx, search_bin_idx), length(search_grid_idx), length(search_bin_idx));
  ave_int = max(cur_frame_vt_data, [], 2);
  all_ave_int(which_frame,:) = ave_int;
  all_ave_int_diff(which_frame,:) = (-1) * (ave_int(2:end) - ave_int(1:end-1)); % smallest (-) intensity derivative grid will be tracked.
end

num_bin = length(search_grid_idx)-1; % each search grid is a bin for this problem
% compute the transition matrix
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
transmat = cell(num_frame-1,1);
for which_frame = 1:(num_frame-1)
    transmat{which_frame} = (tmp_transmat - min(min(tmp_transmat))) ./ (max(max(tmp_transmat)) - min(min(tmp_transmat)));
end

% PRIOR
prior = all_ave_int_diff(1,:); % the value of the first frame

% OBJECT LIKELIHOOD
tmp_obslik = all_ave_int_diff; % derivatives of average pixel intensity
obslik = (tmp_obslik - min(min(tmp_obslik))) ./ (max(max(tmp_obslik)) - min(min(tmp_obslik)));

% viterbi decoding
opt_path = f_viterbi_path_min(prior,obslik,transmat);

% smoothing
track_out.idx_lips = search_grid_idx(opt_path);
pos_lips_x = smooth(grid.center_pt(search_grid_idx(opt_path),1),3,'moving');
pos_lips_y = smooth(grid.center_pt(search_grid_idx(opt_path),2),3,'moving');
track_out.pos_lips = [pos_lips_x pos_lips_y];

%% plot and check
%h=figure; 
%for which_frame = 1:num_frame
%  imagesc(reshape(data_mri2(which_frame,:,:),img_size(1),img_size(2))); colormap(gray); hold on;
%  scatter(track_out.pos_lips(which_frame,1), track_out.pos_lips(which_frame,2),'ro');
%  pause(0.05); hold off;
%end; close(h);


