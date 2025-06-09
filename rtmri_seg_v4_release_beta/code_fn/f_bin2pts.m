function pts = f_bin2pts(bin,grid,init_grid_idx_all,final_grid_idx_all)

% find the pixel coordinates for each bin in the grid lines

% If you have any question, please email to jangwon@usc.edu
% Jangwon Kim
% May 2nd 2014



num_frame = size(bin,1);
num_grid = size(grid.center_pt,1);

pts = zeros(num_frame,num_grid,2);
for which_frame = 1:num_frame
  for which_grid = init_grid_idx_all(which_frame):final_grid_idx_all(which_frame)
    if (isnan(bin(which_frame,which_grid)) == 0)
      pts(which_frame,which_grid,:) = grid.bin_pts(which_grid,round(bin(which_frame,which_grid)),:);
    else
      pts(which_frame,which_grid,:) = NaN;
    end
  end
  pts(which_frame,1:(init_grid_idx_all(which_frame)-1),:) = NaN;
  pts(which_frame,(final_grid_idx_all(which_frame)+1):end,:) = NaN;
end
