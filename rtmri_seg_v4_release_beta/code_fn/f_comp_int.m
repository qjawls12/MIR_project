function grid = f_comp_int(data_mri, grid, opt)

% Compute (smoothed) pixel intensity for each bin

% If you have any question, please email to jangwon@usc.edu
% Jangwon Kim
% May 2nd 2014

img_size = opt.img.img_size;
num_bin = opt.grid.num_bin;
num_frame = size(data_mri,1);
num_grid = size(grid.center_pt,1);

% intensity plot of each gridline
grid.line_intensity = zeros(num_frame,num_grid,num_bin); 
grid.sm_line_intensity = zeros(num_frame,num_grid,num_bin);
for which_frame = 1:num_frame

    for which_grid=1:num_grid

        pix_idx_cur_line_x = reshape(round(grid.bin_pts(which_grid,:,1)),[],1);
        pix_idx_cur_line_y = reshape(round(grid.bin_pts(which_grid,:,2)),[],1);
        
        cur_img = reshape(data_mri(which_frame,:,:),img_size(1),img_size(2));
        cur_img_n = f_norm_min_max(cur_img, 0, 1);
        cur_line_intensity = diag(reshape(cur_img_n(pix_idx_cur_line_y,pix_idx_cur_line_x),num_bin,num_bin));
        grid.line_intensity(which_frame,which_grid,:) = cur_line_intensity;
        
        % smoothing: moving average filter
        sm_cur_line_intensity = smooth(cur_line_intensity,3,'moving');
        grid.sm_line_intensity(which_frame,which_grid,:) = sm_cur_line_intensity;
    end
end

