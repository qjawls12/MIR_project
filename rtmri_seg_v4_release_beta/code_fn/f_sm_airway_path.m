function opt_path_sm = f_sm_airway_path(grid, opt_path, track_out, opt)

% Unsupervised temporal and spatial smoothing of estimated airway path
%   using agglomerative hierarchical clustring (AHC).
% Temporal smoothing:
%   linear interpolation using only nicely-behaving frames
%    (the ones close to the mean) that are selected by AHC.
% Spatial smoothing performs similar interpolation, but across
%   grid lines. 
% The current version performs only temporal smoothing and 
%   minimal spatial smoothing for the edge frames, because
%   spatial smoothing can cause oversmoothing of the airway path,
%   resulting in partially overlapping in the tissue regions.

% If you have any question, please email to jangwon@usc.edu
% Jangwon Kim
% Dec 18 2014

init_grid_idx_all = track_out.idx_lips;
final_grid_idx_all = track_out.idx_larynx;

[num_frame num_grid] = size(opt_path);
st_idx = max(init_grid_idx_all);
e_idx = min(final_grid_idx_all);

num_best_clu = opt.airway.num_best_clu;	% the number of clusters of frames
num_max_clu = opt.airway.num_max_clu;	% the maximum number of clusters (recommended: 2 or 3) 

% [1] temporal smoothing
opt_path_sm1 = opt_path;
for which_grid = st_idx:e_idx

  cur_grid = opt_path(:,which_grid);

  % agglomerative hierarchical clustering of airway path bins
  g_idx_b = f_find_N_best_clu(cur_grid, num_best_clu, num_max_clu);

  g_idx = find(g_idx_b);  % sample index of the biggest cluster 
  b_idx = find(~g_idx_b); % sample index of the other clusters
  x1 = g_idx;
  y1 = cur_grid(x1);
  x2 = (1:num_frame)';

  % interplation
  y2 = interp1(x1, y1, x2, 'linear');

%which_grid
%h=figure; plot(x1,y1); hold on; plot(x2,y2,'r'); hold off; keyboard; close(h);

  % extrapolation for NaN
  % (1) initial NaN frames
  nan_idx = find(isnan([NaN;y2;NaN]));
  idx = min(find((nan_idx(2:end) - nan_idx(1:end-1)) > 1));
  %y2(1:idx-1) = y2(idx);
  y2(1:idx-1) = median(y2(idx:(idx+4)));
  % (2) final NaN frames
  nan_idx = find(isnan([NaN;fliplr(y2')';NaN]));
  idx = min(find((nan_idx(2:end) - nan_idx(1:end-1)) > 1)); 
  %y2((end-idx+1):end) = y2(end-idx+1);
  y2((end-idx+1):end) = median(y2((end-idx-4):(end-idx+1)));

  opt_path_sm1(:,which_grid) = y2;
end



% [2] minimal spatial smoothing for each frame
%  (smoothing on only the grids not smoothed by 
%    above temporal smoothing)
opt_path_sm2 = opt_path_sm1;
for which_frame = 1:num_frame

  st_grid_idx = init_grid_idx_all(which_frame);
  e_grid_idx = final_grid_idx_all(which_frame);
  cur_path = opt_path_sm1(which_frame,st_grid_idx:e_grid_idx)';
  num_grid_cur = e_grid_idx-st_grid_idx+1;

  % agglomerative hierarchical clustering of airway path bins
  g_idx_b = f_find_N_best_clu(cur_path, num_best_clu, num_max_clu);

  tmp_g_idx = zeros(num_grid_cur,1);
  tmp_g_idx((st_idx-st_grid_idx+1):(e_idx-st_grid_idx+1)) = 1;
  g_idx = find(g_idx_b + tmp_g_idx);  % sample index of the biggest cluster 
  b_idx = find(~(g_idx_b + tmp_g_idx)); % sample index of the other clusters
  x1 = g_idx;
  y1 = cur_path(x1);
  x2 = (1:num_grid_cur)';

  % interpolation
  y2 = interp1(x1, y1, x2, 'linear');

  % extrapolation
  % (1) initial NaN frames
  nan_idx = find(isnan([NaN;y2;NaN]));
  idx = min(find((nan_idx(2:end) - nan_idx(1:end-1)) > 1));
  %y2(1:idx-1) = y2(idx);
  y2(1:idx-1) = median(y2(idx:(idx+4)));
  % (2) final NaN frames
  nan_idx = find(isnan([NaN;fliplr(y2')';NaN]));
  idx = min(find((nan_idx(2:end) - nan_idx(1:end-1)) > 1));
  %y2((end-idx+1):end) = y2(end-idx+1);
  y2((end-idx+1):end) = median(y2((end-idx-4):(end-idx+1)));
  opt_path_sm2(which_frame,st_grid_idx:e_grid_idx) = y2;

end

% [3] ignore oversmoothing
%     if smoothed-path bin and original-path bin are very closely located (+-1 bin)
%   & if original-path bin has smaller pixel intensity
opt_path_sm3 = opt_path_sm2;
for which_frame = 1:num_frame

  % bin indexes of non-NaN path
  st_grid_idx = init_grid_idx_all(which_frame);
  e_grid_idx = final_grid_idx_all(which_frame);
  x_idx = st_grid_idx:1:e_grid_idx;

  % find too close original and smoothed paths
  orig = opt_path(which_frame,x_idx);
  sm = round(opt_path_sm2(which_frame,x_idx));
  over_sm_idx = find(abs(orig - sm) <= 2);

  % compare the pixel intensities of original and smoothed paths
  if isempty(over_sm_idx) == 0
    orig_int = diag(reshape(grid.line_intensity(which_frame, x_idx(over_sm_idx), orig(over_sm_idx)),length(over_sm_idx),length(over_sm_idx)));
    sm_int = diag(reshape(grid.line_intensity(which_frame, x_idx(over_sm_idx), sm(over_sm_idx)),length(over_sm_idx),length(over_sm_idx)));

    % recover the original path
    idx = find(orig_int < sm_int);
    opt_path_sm3(which_frame,x_idx(idx)) = opt_path(which_frame,x_idx(idx));
  end
end

opt_path_sm = opt_path_sm3;

