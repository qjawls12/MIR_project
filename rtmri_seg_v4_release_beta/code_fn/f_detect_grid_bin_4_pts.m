function [output_grid, output_bin] = f_detect_grid_bin_4_pts(grid, pts)

min_dist = 1000; % big enough number

num_grid = size(grid.bin_pts,1);
num_bin = size(grid.bin_pts,2);

for which_grid = 1:num_grid
  for which_bin = 1:num_bin
    % center
    cur_dist = sum((pts - reshape(grid.bin_pts(which_grid,which_bin,:),1,2)).^2);
    if min_dist > cur_dist;
      output_grid = which_grid;
      output_bin = which_bin;
      min_dist = cur_dist;
    end
  end
end
