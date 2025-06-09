function [bin_sm,bin_i] = f_sm_bin(bin, grid, opt)

% generates two outputs
%   bin_sm: smoothed matrix
%   bin_i: interpolated matrix for NaNs

% If you have any question, please email to jangwon@usc.edu
% Jangwon Kim
% Dec 18 2014

% Assign simpler variable name
win_span = opt.bnd.win_span_sm;
sm_param = opt.bnd.sm_param;
dct_order = opt.bnd.dct_order;

num_best_clu = opt.bnd.phar.num_best_clu; % the number of clusters of frames
num_max_clu = opt.bnd.phar.num_max_clu;  % the maximum number of clusters
sm_param_phar = opt.bnd.phar.sm_param;
dct_order_phar = opt.bnd.phar.dct_order;

num_frame = size(bin,1);
num_grid = size(bin,2);
num_bin = size(grid.bin_pts,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% interpolation on the grids inside the vocal tract for each frame.
% - filling up NaN grid inside the vocal tract
% - points outside the vocal tract still have NaNs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bin_i = bin;
for which_frame = 1:num_frame
  nonan_idx = find(~isnan(bin(which_frame,:)));
  if isempty(nonan_idx) == 0
    nonan_min_idx = min(nonan_idx);
    nonan_max_idx = max(nonan_idx);
    x1=nonan_idx;
    y1=bin(which_frame,x1);
    x2=nonan_min_idx:nonan_max_idx;
    y2=interp1(x1,y1,x2,'linear');
    bin_i(which_frame,x2)=y2;
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% smoothing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Spatial smoothing for the region until alveolar ridge grid
%  & constrain to the palate landmark point

% (a) generates the palate boundary points
% find the closest gridline points for the center, left-most,
%   right-most points of the mpal_samp
[mpal_c_idx_grid,mpal_c_idx_bin] = f_detect_grid_bin_4_pts(grid, grid.mpal);
[mpal_l_idx_grid,mpal_l_idx_bin] = f_detect_grid_bin_4_pts(grid, [grid.mpal - [win_span 0]]);
[mpal_r_idx_grid,mpal_r_idx_bin] = f_detect_grid_bin_4_pts(grid, [grid.mpal + [win_span 0]]);
x1 = [mpal_l_idx_grid;mpal_c_idx_grid;mpal_r_idx_grid];
y1 = [mpal_l_idx_bin; mpal_c_idx_bin; mpal_r_idx_bin];
mpal_grid = mpal_l_idx_grid:1:mpal_r_idx_grid;
mpal_bins = interp1(x1,y1,mpal_grid,'linear');

% (b) find the lower or mean points between the palate boundary points
%       and the outer boundary, then perform smoothing
bin_sm = bin_i;
for which_frame = 1:num_frame
  nonan_idx = find(~isnan(bin_i(which_frame,:)));
  [x1_min,idx_min] = max([nonan_idx(1) mpal_grid(1)]);
  [x1_max,idx_max] = min([nonan_idx(end) mpal_grid(end)]);
  mpal_grid_min = find(mpal_grid == x1_min);
  mpal_grid_max = find(mpal_grid == x1_max);
  mpal_grid_tmp = mpal_grid(mpal_grid_min:mpal_grid_max);
  mpal_bins_tmp = mpal_bins(mpal_grid_min:mpal_grid_max);
  x1 = x1_min:x1_max;
  y1 = bin_i(which_frame,x1);

  y_mpal_l_r = bin_i(which_frame,mpal_grid_tmp);
  switch opt.tmp.pal
    case 1 % find the lower boundary points for the mpal region
        y_mpal_opt = min([y_mpal_l_r;mpal_bins_tmp]);
    case 2 % find the mean boundary points for the mpal region
        y_mpal_opt = mean([y_mpal_l_r;mpal_bins_tmp],1);
    case 3 % use the original boundary points without any constraint
        y_mpal_opt = y_mpal_l_r;
  end

  % update the lower points
  y1 = y_mpal_opt;

  % smoothing for the palatal region
  y2 = smooth(x1,y1,0.4,'rlowess');
  bin_sm(which_frame,x1) = y2;

  %% smoothing from the 1st grid to the end of hard palate
  %x1 = 1:x1(end);
  %y3 = bin_sm(which_frame,x1);
  %y4 = smooth(x1,y3,0.1,'rlowess');
  %bin_sm(which_frame,x1) = y4;
end

% (d) smoothing pharyngeal wall (bottom half from grid.mpal(2) to grid.larynx(2))
bin_sm1 = bin_sm;
switch opt.tmp.phar
  case 1 % AHC based interpolation
    % find index of grid closest to the vertical middle point of grid.mpal and grid.larynx
    [grid_phar_top_idx,~] = f_detect_grid_bin_4_pts(grid, [grid.larynx(1) mean([grid.mpal(2) grid.larynx(2)])]);
    for which_frame = 1:num_frame

      % choose the lower half of the boundary points from mpal point
      grid_topLarynx_idx = max(find(~isnan(bin_sm(which_frame,:))));
      x_I = grid_phar_top_idx:grid_topLarynx_idx; % grid lines of interest
      data = bin_sm(which_frame, x_I);

      % smoothing of pharyngeal region
      bin_sm(which_frame,x_I) = medfilt1(data,5);

    end % for

  case 2 % only edge (laryngeal region) smoothing
    for which_frame = 1:num_frame
      grid_topLarynx_idx = max(find(~isnan(bin_sm(which_frame,:))));
      x_I = (grid_topLarynx_idx-3):1:grid_topLarynx_idx;
      data = bin_sm(which_frame, x_I)';
      y3 = [data;flipud(data)];
      y3_med = repmat(median(y3),length(y3),1);
      bin_sm(which_frame, x_I) = y3_med(1:length(x_I));
    end
end % switch

% find the grid line index closest to grid.mlab
[grid_mlab_idx,~] = f_detect_grid_bin_4_pts(grid, grid.mlab);

% (e) DCT smoothing in [mlab non-NaN-final], 
%       followed by constraining the range of bin: [1 num_bin]
bin_sm2 = bin_sm;
for which_frame = 1:num_frame
  nonan_idx = find(~isnan(bin_i(which_frame,:)));
  % bin_i;
  data = bin_i(which_frame,nonan_idx);
  data(find(data < 1)) = 1;
  data(find(data > num_bin)) = num_bin;
  bin_i(which_frame,nonan_idx) = data;
  
  % bin_sm;
  nonan_idx = find(~isnan(bin_sm(which_frame,:)));
  x = max([grid_mlab_idx nonan_idx(1)]):1:nonan_idx(end);
  data = bin_sm(which_frame,x);
  data2 = dct_sm(data', sm_param, dct_order); % DCT smoothing
  data2(find(data2 < 1)) = 1;
  data2(find(data2 > num_bin)) = num_bin;
  bin_sm(which_frame,x) = data2;
end

%figure;
%for which_frame = 1:num_frame
%  plot(bin_i(which_frame,:),'k'); hold on; % interpolation
%  plot(bin_sm1(which_frame,:),'b');     % processing palatal region
%  plot(bin_sm2(which_frame,:),'g');     % processing pharyngeal region
%  plot(bin_sm(which_frame,:),'r'); hold off; % processing DCT
%  axis([1 num_grid 1 num_bin]);
%  title(which_frame);
%  %pause(1/opt.img.fr_image);
%  pause;
%end

