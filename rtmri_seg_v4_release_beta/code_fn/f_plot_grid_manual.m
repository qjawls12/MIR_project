function f_plot_grid_manual(grid, opt, path)

% Load a reference MR image
list = dir(fullfile(path.mri_data_dir, '*.mat'));
ref_video_path = fullfile(path.mri_data_dir, list(1).name);
data_mri = f_load_data_from_mat(ref_video_path);
img_ref = reshape(data_mri(opt.grid.ref_img_idx,:),opt.img.img_size(1),opt.img.img_size(2));

% std image
img_std = reshape(std(data_mri),opt.img.img_size(1),opt.img.img_size(2));

% Assign simpler variable names
outer_dist = opt.airway.outer_dist;
inner_dist = opt.airway.inner_dist;
num_bin = opt.grid.num_bin;

% Parameters
num_grid = size(grid.center_pt,1);
outer_dist_bin = outer_dist * 2;
inner_dist_bin = inner_dist * 2;

% (1) plot grid lines
figure('Position', [100 100 400 400]);
imagesc(img_ref); colormap(gray); hold on;
for which_grid=1:num_grid
  scatter(grid.center_pt(which_grid,1),grid.center_pt(which_grid,2),'g.');
  plot([grid.line_lb(which_grid,1) grid.line_rt(which_grid,1)],[grid.line_lb(which_grid,2) grid.line_rt(which_grid,2)],'c');
end
%scatter(grid.mlab(1),grid.mlab(2),'y.');
%scatter(grid.larynx(1),grid.larynx(2),'y.');
%scatter(grid.mpal(1),grid.mpal(2),'y.');
title('Constructed grid lines');
hold off;
%disp('Change opt.grid.gwidth for different length of grid lines...');

% (2) plot the regions for estimating airway-path
if (isempty(outer_dist) == 0) || (isempty(inner_dist) == 0)
  figure('Position', [100 100 800 400]);
  % in a reference image
  subplot(1,2,1)
  imagesc(img_ref); colormap(gray); hold on;
  for which_grid=1:num_grid
    if isempty(outer_dist) == 0
      idx_outer_bnd = min(find(pdist2(reshape(grid.bin_pts(which_grid,(round(num_bin*0.5)+1):1:end,:),[],2),grid.center_pt(which_grid,:)) > outer_dist_bin) + round(num_bin*0.5));
    end
    if isempty(inner_dist) == 0
      idx_inner_bnd = max(find(pdist2(reshape(grid.bin_pts(which_grid,1:1:(round(num_bin*0.5)+1),:),[],2),grid.center_pt(which_grid,:)) > inner_dist_bin));
    end
    plot([grid.bin_pts(which_grid,idx_outer_bnd,1) grid.bin_pts(which_grid,idx_inner_bnd,1)], ...
         [grid.bin_pts(which_grid,idx_outer_bnd,2) grid.bin_pts(which_grid,idx_inner_bnd,2)], 'c');
  end
  title('Region of airway-path estimation (in ref image)');
  hold off;
  % in a std image
  subplot(1,2,2)
  imagesc(img_std); colormap(gray); hold on;
  for which_grid=1:num_grid
    if isempty(outer_dist) == 0
      idx_outer_bnd = min(find(pdist2(reshape(grid.bin_pts(which_grid,(round(num_bin*0.5)+1):1:end,:),[],2),grid.center_pt(which_grid,:)) > outer_dist_bin) + round(num_bin*0.5));
    end
    if isempty(inner_dist) == 0
      idx_inner_bnd = max(find(pdist2(reshape(grid.bin_pts(which_grid,1:1:(round(num_bin*0.5)+1),:),[],2),grid.center_pt(which_grid,:)) > inner_dist_bin));
    end
    plot([grid.bin_pts(which_grid,idx_outer_bnd,1) grid.bin_pts(which_grid,idx_inner_bnd,1)], ...
         [grid.bin_pts(which_grid,idx_outer_bnd,2) grid.bin_pts(which_grid,idx_inner_bnd,2)], 'c');
  end
  %scatter(grid.mlab(1),grid.mlab(2),'y.');
  %scatter(grid.larynx(1),grid.larynx(2),'y.');
  %scatter(grid.mpal(1),grid.mpal(2),'y.');
  title('Region of airway-path estimation (in std image)');
  hold off;
end
%disp('Change opt.airway.outer_dist and opt.airway.inner_dist for different region of airway-path estimation...');
