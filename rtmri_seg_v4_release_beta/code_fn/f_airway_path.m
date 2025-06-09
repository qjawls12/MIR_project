function opt_path = f_airway_path(data_mri, grid, track_out, opt)

% Estimate the optimal airway path line using the Viterbi algorithm.
% Currently, observation score can be smoothed by one of 2 options:
%   option 1: computing the mean of quantiles of neighbor pixel 
%               intensity for the corresponding pixel
%   option 0: no smoothing
% The smoothing process takes the most time in this process.
% The current version gives more cost on transition matrix if going to the 
%   point higher than the highest palate landmark point.

% Assign simpler variable name
grid_line_idx_larynx = track_out.idx_larynx;
grid_line_idx_lips = track_out.idx_lips;
num_bin = opt.grid.num_bin;
sigmoid_param = opt.airway.sigmoid_param_air;
sigmoid_param_lip = opt.airway.sigmoid_param_air_lip;
trans_w = opt.airway.trans_w;		% weight on smoothness of airway path curve
obslik_sm_frame_span = opt.airway.obslik_sm_frame_span;	% smoothing using X frames (X for one side)
obslik_sm_grid_span = opt.airway.obslik_sm_grid_span;	% smoothing using Y grids (Y for one side)
obslik_sm_bin_span = opt.airway.obslik_sm_bin_span;	% smoothing using Z bins (Z for one side)
outer_dist = opt.airway.outer_dist;
inner_dist = opt.airway.inner_dist;

outer_dist_bin = outer_dist * 2;
inner_dist_bin = inner_dist * 2;

num_frame = size(data_mri,1);
num_grid = size(grid.center_pt,1);

opt_path = zeros(num_frame,num_grid);

% compute transition matrix (between bins of adjacent grid lines)
transmat = cell(num_grid-1,1);
for which_grid = 1:(num_grid-1)
    cur_bin_pts = reshape(grid.bin_pts(which_grid,:,:),num_bin,2);
    next_bin_pts = reshape(grid.bin_pts(which_grid+1,:,:),num_bin,2);
    D = pdist2(cur_bin_pts,next_bin_pts,'euclidean');
    D_n = (D - min(min(D))) ./ (max(max(D)) - min(min(D)));
    D_w = f_sigmoid_warping(D_n,sigmoid_param);
    D_w1 = D_w;

    % make the distance score sharper and narrower
    %   from the initial grid to the lip region
    if (grid.center_pt(which_grid,1) < (grid.mlab(1)+5))
        D_w = f_sigmoid_warping(D_n,sigmoid_param_lip);
    end
    D_w2 = D_w;

    % make the estimated cost higher outside of estimation region
    idx_outer = []; idx_inner = [];
    if isempty(outer_dist) == 0
      idx_outer = find(pdist2(reshape(grid.bin_pts(which_grid,(round(num_bin*0.5)+1):1:end,:),[],2),grid.center_pt(which_grid,:)) > outer_dist_bin) + round(num_bin*0.5);
    end
    if isempty(inner_dist) == 0
      idx_inner = find(pdist2(reshape(grid.bin_pts(which_grid,1:1:(round(num_bin*0.5)+1),:),[],2),grid.center_pt(which_grid,:)) > inner_dist_bin);
    end
    idx = [idx_outer;idx_inner];
    if isempty(idx) == 0
      D_w(:,idx) = 1;
    end
    D_w3 = D_w;

    transmat{which_grid} = f_norm_min_max(D_w,0,trans_w);
end


init_grid_idx_all = zeros(num_frame,1);
final_grid_idx_all = zeros(num_frame,1);

% compute cost matrix
obslik_all = zeros(num_frame,num_grid,num_bin);
for which_frame = 1:num_frame

  % find the initial and final grid lines for current frame
  init_grid_idx = grid_line_idx_lips(which_frame);
  final_grid_idx = grid_line_idx_larynx(which_frame);

  if isempty(init_grid_idx) == 1, init_grid_idx = 1; end;
  if (init_grid_idx < 1), init_grid_idx = 1; end;

  if isempty(final_grid_idx) == 1, final_grid_idx = num_grid; end;
  if (final_grid_idx > num_grid), final_grid_idx = num_grid; end;
  
  init_grid_idx_all(which_frame) = init_grid_idx;
  final_grid_idx_all(which_frame) = final_grid_idx;

  % PRIOR
  %% (1) prior: equal
  %prior = repmat(1/num_bin,1,num_bin);
  % (2) prior: horizontal distance from grid.larynx point to each bin of
  %              the first grid
  pix_idx_cur_line_x = (linspace(grid.line_lb(init_grid_idx,1),grid.line_rt(init_grid_idx,1),num_bin));
  pix_idx_cur_line_y = (linspace(grid.line_lb(init_grid_idx,2),grid.line_rt(init_grid_idx,2),num_bin));
  dist_mat = pdist2([pix_idx_cur_line_x' pix_idx_cur_line_y'],grid.larynx,'euclidean');
  prior = f_norm_min_max(dist_mat,0,1)';

  % OBJECT LIKELIHOOD
  int_mat = reshape(grid.line_intensity(which_frame,:,:), num_grid, num_bin);
  obslik = f_norm_min_max(int_mat,0,1);
  
  % For obslik, use the grid lines from larynx to lips only
  if final_grid_idx < num_grid, 
      obslik(((final_grid_idx+1):end),:) = 0; 
  end;
  if init_grid_idx > 1, 
      obslik(1:(init_grid_idx-1),:) = 0;
  end;
  
  obslik_all(which_frame,:,:) = obslik;
end

% Observation score smoothing
tic1 = tic;
obslik_sm_all = ones(size(obslik_all));
% Option 1: smoothing of obslik_all using neighbors
if opt.airway.obj_sm == 1
  disp('observation score smoothing starts... ');
  siz = ([obslik_sm_frame_span obslik_sm_grid_span obslik_sm_bin_span]*2)+1;
  obslik_sm_all_Q1 = quanfilt3(obslik_all,siz,0.75);
  obslik_sm_all_Q2 = quanfilt3(obslik_all,siz,0.25);
  obslik_sm_all = reshape(mean([reshape(obslik_sm_all_Q1,[],1) reshape(obslik_sm_all_Q2,[],1)],2),size(obslik_all));
% Option 0: exclude smoothing
elseif opt.airway.obj_sm == 0
  obslik_sm_all = obslik_all;
else
  error('opt.airway.obj_sm is not correct....');
end
toc(tic1)
 
% Viterbi decoding 
disp('viterbi decoding starts... ');
for which_frame = 1:num_frame

  final_grid_idx = final_grid_idx_all(which_frame);  
  obslik = reshape(obslik_sm_all(which_frame,:,:),num_grid,num_bin);
  
  % give less cost if the pixel is close to the lip landmark point
  final_grid_pts = reshape(grid.bin_pts(final_grid_idx,:,:),num_bin,2);
  for i=final_grid_idx:1:num_grid
      tmp = f_norm_min_max(pdist2(final_grid_pts(:,2),grid.mlab(:,2),'euclidean'),0,1);
      obslik(i,:) = f_sigmoid_warping(tmp,[10 0.3]);
  end    
  
  path = f_viterbi_path_min(prior,obslik,transmat);
  opt_path(which_frame,:) = path';
end

