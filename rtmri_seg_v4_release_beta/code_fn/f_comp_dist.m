function [vt_d,vt_dL] = f_comp_dist(bin_lb, bin_rt, pts_lb, pts_rt)

% Compute distance function from the estimated tissue-airway boundary points
% This version computes the Euclidean distance between the upper and 
%   lower boundaries in each grid line.
% outputs:
%  vt_d: vocal tract boundary distance for each grid
%  vt_dL: vt_d until the first local minimum distance point, 
%           from the front-most grid line, in the lip region

% If you have any question, please email to jangwon@usc.edu
% Jangwon Kim
% Dec 23 2014


[num_frame, num_grid] = size(bin_lb);
num_lip_grid_srch = round(num_grid/10);

% compute the Euclidean distance between boundaries 
vt_d = zeros(num_frame,num_grid);
init_vt_idx_all = zeros(num_frame,1);
for which_frame = 1:num_frame
  % find noNaN index
  nonan_lb_b = ~isnan(bin_lb(which_frame,:));
  nonan_rt_b = ~isnan(bin_rt(which_frame,:));
  nonan_idx = find(nonan_lb_b .* nonan_rt_b);
  nan_idx = find(~(nonan_lb_b .* nonan_rt_b));
  
  % (A) compute distance for each grid line
  vt_d(which_frame,nonan_idx) = sqrt(sum(reshape((pts_lb(which_frame,nonan_idx,:) - pts_rt(which_frame,nonan_idx,:)).^2,[],2),2));
  vt_d(which_frame,nan_idx) = NaN;

  % find the first local minimum point within lip region
  data_fl = (-1) * (vt_d(which_frame,nonan_idx(1:num_lip_grid_srch)));
  data_fl = [(min(data_fl)-10) data_fl];
  data = smooth(data_fl,3);
  [~,tmp_loc] = findpeaks(data);
  if isempty(tmp_loc) == 1, tmp_loc = num_lip_grid_srch; end;
  x = 1:tmp_loc(1);
  [~,loc] = max(data_fl(x));
  while(1) 
    if loc == 1, break; end; 
    if (data_fl(loc) <= data_fl(loc-1)) 
      loc = loc - 1; 
    else 
      break; 
    end 
  end 
  init_vt_idx = (nonan_idx(1) + loc) - 2;
  init_vt_idx_all(which_frame) = init_vt_idx;
end

% interpolate NaN in init_vt_idx_all
x1 = find(~isnan(init_vt_idx_all));
y1 = init_vt_idx_all(x1);
x2 = (1:num_frame)';
y2 = interp1(x1,y1,x2,'linear');
% extrapolation
% (1) init NaN frames
nan_idx = find(isnan([NaN;y2;NaN]));
idx = min(find((nan_idx(2:end) - nan_idx(1:end-1)) > 1));
y2(1:idx-1) = y2(idx);
% (2) final NaN frames
nan_idx = find(isnan([NaN;fliplr(y2')';NaN]));
idx = min(find((nan_idx(2:end) - nan_idx(1:end-1)) > 1));
y2((end-idx+1):end) = y2(end-idx+1);

% in case of out-of-boundary
init_vt_idx_c = min([y2 repmat(num_grid,num_frame,1)],[],2);

% compute vt_dL
warning('off','all');
vt_dL = zeros(size(bin_lb));
for which_frame = 1:num_frame
  vt_dL(which_frame,:) = vt_d(which_frame,:);
  vt_dL(which_frame,1:(init_vt_idx_c(which_frame)-1)) = NaN;
end
