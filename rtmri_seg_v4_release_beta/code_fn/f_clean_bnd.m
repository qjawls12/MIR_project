function [bin_lb_c, bin_rt_c] = f_clean_bnd(bin_lb, bin_rt, vt_dL)

% Removing the noisy edge boundaries based on
%   estimated vocal tract distance parameter (ending at the
%   lip point)
% Replace outer and inner boundaries to their mean when they cross.

% If you have any question, please email to jangwon@usc.edu
% Jangwon Kim
% April 12 2015


[num_frame num_grid] = size(vt_dL);

% Remove boundaries outside of the vocal tract
bin_lb_c = bin_lb;
bin_rt_c = bin_rt;
for iFrame = 1:num_frame
  nan_idx = isnan(vt_dL(iFrame,:));
  bin_lb_c(iFrame,nan_idx) = NaN;
  bin_rt_c(iFrame,nan_idx) = NaN;
end

% Replace outer and inner boundaries to their mean when they cross.
for iFrame = 1:num_frame
  idx = find(bin_lb_c(iFrame,:) > bin_rt_c(iFrame,:)); % if cross
  sub_bins = mean([bin_lb_c(iFrame,idx);bin_rt_c(iFrame,idx)],1);
  bin_lb_c(iFrame,idx) = sub_bins;
  bin_rt_c(iFrame,idx) = sub_bins;
end
