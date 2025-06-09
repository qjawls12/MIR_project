function grid = f_grid_manual(opt, path)

grid_pnt_sel_frame_idx = opt.grid.ref_img_idx;
img_size = opt.img.img_size;
gint = opt.grid.gint;
sm_param = opt.grid.sm_param;
dct_order = opt.grid.dct_order;
num_grid_lip_prot = opt.grid.num_grid_lip_prot;
num_grid_larynx_motion = opt.grid.num_grid_larynx_motion;

% Load a reference MR image
list = dir(fullfile(path.mri_data_dir, '*.mat'));
ref_video_path = fullfile(path.mri_data_dir, list(1).name);
data_mri = f_load_data_from_mat(ref_video_path);
img_ref = reshape(data_mri(opt.grid.ref_img_idx,:),opt.img.img_size(1),opt.img.img_size(2));

% std image
img_std = reshape(std(data_mri),opt.img.img_size(1),opt.img.img_size(2));

% manually draw the center line for the grid
h=figure('Position',[100 100 500 500]);
imagesc(img_std); colormap(gray);
disp('From the lips to below the larynx, drag a line for centers of grid lines (pressing the left button of your mouse)...');
h_hand = imfreehand('Closed',false);
tmp2 = getPosition(h_hand);
close(h);

% manually choose 2 landmark points (larynx and lips)
h=figure('Position',[100 100 500 500]);
imagesc(img_ref); colormap(gray);
fprintf( '\n   Select the lowest point of the upper lip');
grid.mlab       = ginput(1);
fprintf( '\n   Select the top of the hard palate');
grid.mpal       = ginput(1);
fprintf( '\n   Select the point near the pharyngeal wall (horizontally) and the top of the larynx (vertically)');
grid.larynx     = ginput(1);
fprintf( '\n   All anatomical landmarks are acquired...\n\n');
close(h);

% choose equally spaced points
tmp3 = [];
p_init = tmp2(1,:);
p_end = tmp2(end,:);
num_p = size(tmp2,1);
tmp_tmp = tmp2;
tmp3 = [tmp3; p_init]; % 1st point (larynx)
cur_p = p_init;
while 1
  % choose (gint/2)-pt-dist points on the manually drawn line
  [val,idx] = sort(abs(sqrt(sum((repmat(cur_p,size(tmp_tmp,1),1) - tmp_tmp).^2,2)) - (gint/2)));
  if (max(val) >= (gint/2))
    cur_p = tmp_tmp(idx(1),:);
    tmp3 = [tmp3; cur_p];
    tmp_tmp(1:idx(1),:) = [];
  else
    break;
  end
end
tmp3 = [tmp3; tmp2(end,:)];

% DCT smoothing
tmp4 = dct_sm(tmp3, sm_param, dct_order);
tmp4 = [tmp3(1,:); tmp4; tmp3(end,:)];

% choose equally spaced points
tmp5 = [];
p_init = tmp4(1,:);
p_end = tmp4(end,:);
num_p = size(tmp4,1);
tmp_tmp = tmp4;
tmp5 = [tmp5; p_init];
cur_p = p_init;
while 1
  % choose gint-pt-dist points on the manually drawn line
  [val,idx] = sort(abs(sqrt(sum((repmat(cur_p,size(tmp_tmp,1),1) - tmp_tmp).^2,2)) - gint));
  if (max(val) >= gint) && (length(idx) > 1)
    nei1 = tmp_tmp(idx(1),:);
    nei2 = tmp_tmp(idx(2),:);
    tmp_tmp(1:idx(1),:) = [];

    w_nei1 = abs(norm(nei1 - tmp5(end,:)) - gint);
    w_nei2 = abs(norm(nei2 - tmp5(end,:)) - gint);
    w_nei1_n = w_nei1 / (w_nei1 + w_nei2);
    w_nei2_n = w_nei2 / (w_nei1 + w_nei2);
    next_p = w_nei1_n * nei2 + w_nei2_n * nei1;
    tmp5 = [tmp5; next_p];
    cur_p = next_p;
  else
    % for the final point
    final_p = tmp5(end,:) + (tmp5(end,:)-tmp5(end-1,:));
    tmp5 = [tmp5; final_p];
    break;
  end
end
grid.center_pt = tmp5;

%% plot the center of grid lines
%figure; 
%imagesc(img_ref); colormap(gray); hold on;
%scatter(tmp2(:,1),tmp2(:,2),'b');
%scatter(tmp5(:,1),tmp5(:,2),'r.'); hold off;
%legend('manual','grid center');

% choose points between larynx and lip landmark points
idx = find((grid.center_pt(:,1) - grid.mlab(:,1)) < 0);
grid.center_pt(idx,:) = [];
idx = find((grid.center_pt(:,2) - grid.larynx(:,2)) > 0);
grid.center_pt(idx,:) = [];

% Determine the centers of the lip protrusion grid
%       (extension from the final point) 
for which_grid=1:num_grid_lip_prot
  new_p = grid.center_pt(1,:) + (grid.center_pt(1,:) - grid.center_pt(2,:));
  grid.center_pt = [new_p;grid.center_pt];
end

% Determine the centers of the larynx motion grid
%       (extention from the initial point)
for which_grid=1:num_grid_larynx_motion
  new_p = grid.center_pt(end,:) + (grid.center_pt(end,:) - grid.center_pt(end-1,:));
  grid.center_pt = [grid.center_pt;new_p];
end

figure('Position', [100 100 800 400]);
subplot(1,2,1)
imagesc(img_std); colormap(gray); hold on; 
scatter(grid.center_pt(:,1),grid.center_pt(:,2),'y.');
hold off;
title('Center points of grid lines');
subplot(1,2,2)
imagesc(img_ref); colormap(gray); hold on;
scatter(grid.center_pt(:,1),grid.center_pt(:,2),'y.');
plot(grid.mlab(1),grid.mlab(2),'r.','MarkerSize',25);
plot(grid.mpal(1),grid.mpal(2),'r.','MarkerSize',25);
plot(grid.larynx(1),grid.larynx(2),'r.','MarkerSize',25);
hold off;
title('Center points (yellow) of grid lines and landmarks (red)');

% EOF

