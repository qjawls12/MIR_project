function [pts_c, vtl] = f_det_center(pts_lb, pts_rt, opt)

% Determine the center points between outer and inner 
%   tissue-airway boundaries
% Then, compute geodesic distance from the lip to the larynx

num_frames = size(pts_lb,1);
num_grid = size(pts_lb,2);


% Determine the center point for each grid line
pts_c = zeros(size(pts_lb));
vtl = zeros(num_frames,num_grid);
for iFrame = 1:num_frames

  nan_idx = find(isnan(pts_lb(iFrame,:,1)));
  nonan_idx = find(~isnan(pts_lb(iFrame,:,1)));

  pts_lb_x = pts_lb(iFrame,nonan_idx,1);
  pts_c(iFrame,nonan_idx,1) = mean([pts_lb(iFrame,nonan_idx,1)' pts_rt(iFrame,nonan_idx,1)'],2);
  pts_c(iFrame,nonan_idx,2) = mean([pts_lb(iFrame,nonan_idx,2)' pts_rt(iFrame,nonan_idx,2)'],2);
  pts_c(iFrame,nan_idx,:) = NaN;

  pts_c_nonan=reshape(pts_c(iFrame,nonan_idx,:),[],2);
  vtl(iFrame,nonan_idx(end))=vtl(iFrame,nonan_idx(end-1));
  nonan_idx(end) = [];
  vtl(iFrame,nonan_idx) = sqrt(sum((pts_c_nonan(2:end,:) - pts_c_nonan(1:(end-1),:)).^2,2));
  vtl(iFrame,nan_idx) = NaN;

end


