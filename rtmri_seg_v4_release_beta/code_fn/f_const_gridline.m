function grid = f_const_gridline(grid, opt)

% (i) Construct grid lines from manually drawn centers for the grid lines
% (ii) Determine the coordinate of each bin

% If you have any question, please email to jangwon@usc.edu
% Jangwon Kim
% Dec 19 2014

% Assign simpler variable names
gwidth = opt.grid.gwidth;
num_bin = opt.grid.num_bin;

grid.line_lb=[]; % left or bottom
grid.line_rt = []; % right or top

num_grid = size(grid.center_pt,1);

% add extra points, one for each side
num_extra_pts = round(num_grid/5);
extra_pt_lips = flipud(2*repmat(grid.center_pt(end,:),num_extra_pts,1) - grid.center_pt((end-num_extra_pts):(end-1),:));
extra_pt_larynx = flipud(2*repmat(grid.center_pt(1,:),num_extra_pts,1) - grid.center_pt(2:(num_extra_pts+1),:));
grid_c_pt = [extra_pt_larynx; grid.center_pt; extra_pt_lips];

% rotation matrices
outer_m = [0 1; (-1) 0];
inner_m = [0 (-1); 1 0];

% determine outer and inner boundaries
win_len = num_extra_pts;
for iGrid = 1:num_grid
  tmp = (grid_c_pt(iGrid+win_len*2,:) - grid_c_pt(iGrid,:))';
  tmp_n = tmp / norm(tmp);
  outer_p = grid.center_pt(iGrid,:)' + (outer_m * tmp_n) * (gwidth/2);
  inner_p = grid.center_pt(iGrid,:)' + (inner_m * tmp_n) * (gwidth/2);
  grid.line_lb = [grid.line_lb; inner_p'];
  grid.line_rt = [grid.line_rt; outer_p'];
end

% each bin point coordinate of grid line
grid.bin_pts = zeros(num_grid,num_bin,2);
for which_grid = 1:num_grid
    grid.bin_pts(which_grid,:,1) = linspace(grid.line_lb(which_grid,1),grid.line_rt(which_grid,1),num_bin);
    grid.bin_pts(which_grid,:,2) = linspace(grid.line_lb(which_grid,2),grid.line_rt(which_grid,2),num_bin);
end

