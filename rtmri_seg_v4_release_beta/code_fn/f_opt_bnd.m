function bnd_out = f_opt_bnd(data_mri, grid, opt_path, track_out, bnd_out, opt)

% Find the boundary bins for each grid
% Find the first bin whose pixel intensity > threshold in each grid line
%   by searching from the estimated airway path point

% If you have any question, please email to jangwon@usc.edu
% Jangwon Kim
% Dec 18 2014

% Assign simpler variable name
init_grid_idx_all = track_out.idx_lips;
final_grid_idx_all = track_out.idx_larynx;
bnd_thld = opt.bnd.bnd_thld;

num_frame = size(data_mri,1);
num_bin = size(grid.bin_pts,2);
num_grid = size(grid.center_pt,1);

line_lb_bnd_bin = zeros(num_frame,num_grid);
line_rt_bnd_bin = zeros(num_frame,num_grid);

% compute the threshold value for each grid line
qt_thld_int = zeros(num_grid,1);
for i=1:num_grid
    pixel_intensity_data = reshape(grid.sm_line_intensity(:,i,:),1,[]);
    ub = max(pixel_intensity_data);
    lb = min(pixel_intensity_data);
    qt_thld_int(i) = max([(((ub-lb) * bnd_thld) + lb),(bnd_thld/2)]);
end

for which_frame = 1:num_frame
    for which_grid = init_grid_idx_all(which_frame):final_grid_idx_all(which_frame)
        
        % median filter for intensity of each grid
        lin_int = reshape(grid.sm_line_intensity(which_frame,which_grid,:),[],1);

        % (1) Left-bottom (inner boundary)
        left_bins_int = lin_int(1:round(opt_path(which_frame,which_grid)));
        cand_bin_idx = find(left_bins_int > qt_thld_int(which_grid));
        if (isempty(left_bins_int) == 1)    % opt path is out of the current grid line range
            line_lb_bnd_bin(which_frame,which_grid) = 1; % assuming that gridline system covers tissue-airway boundaries
        else
            if (isempty(cand_bin_idx) == 1)
                line_lb_bnd_bin(which_frame,which_grid) = NaN; % open space
            else
                line_lb_bnd_bin(which_frame,which_grid)=max(cand_bin_idx);
            end
        end

        % (2) Right-top (outer boundary)
        right_bins_int = lin_int(round(opt_path(which_frame,which_grid)):end);
        cand_bin_idx = find(right_bins_int > qt_thld_int(which_grid));
        if isempty(right_bins_int) == 1     % opt path is out of the current grid line range
              line_rt_bnd_bin(which_frame,which_grid) = num_bin; % assuming that gridline system covers tissue-airway boundaries
        else
            if (isempty(cand_bin_idx) == 1)
                line_rt_bnd_bin(which_frame,which_grid) = NaN; % open space
            else
                line_rt_bnd_bin(which_frame,which_grid)=min(cand_bin_idx) + round(opt_path(which_frame,which_grid)) - 1;
            end
        end
    end

  %% check determined boundary point for a grid line
  %figure; imagesc(reshape(data_mri(which_frame,:,:),68,68)); colormap(gray); hold on;
  %tmp=65; tmp2 = reshape(grid.bin_pts(tmp,line_rt_bnd_bin(which_frame,tmp),:),1,2); scatter(tmp2(1),tmp2(2),'r.');

end

%% plot pixel intensity for each frame and each grid
%keyboard;
%which_frame = 50;
%which_grid = 9;
%lin_int = reshape(grid.sm_line_intensity(which_frame,which_grid,:),[],1);
%lb = line_lb_bnd_bin(which_frame,which_grid);
%rb = line_rt_bnd_bin(which_frame,which_grid);
%ct = round(opt_path(which_frame,which_grid));
%h=figure; plot(lin_int,'k'); hold on;
%%line([lb lb]-0.4,[0 max(lin_int)],'Color','r','LineWidth',2);
%line([ct ct],[0 max(lin_int)],'Color','b','LineWidth',2);
%%line([rb rb]+0.4,[0 max(lin_int)],'Color','g','LineWidth',2);
%hold off;
%axis([1 num_bin 0 max(lin_int)*1.1]);


% if bin == 0, put NaN
for which_frame = 1:num_frame
    for which_grid = 1:num_grid
        if (line_lb_bnd_bin(which_frame,which_grid)) == 0
            line_lb_bnd_bin(which_frame,which_grid) = NaN;
        end        
        if (line_rt_bnd_bin(which_frame,which_grid)) == 0
            line_rt_bnd_bin(which_frame,which_grid) = NaN;
        end
    end
end

bnd_out.bin_lb = line_lb_bnd_bin;
bnd_out.bin_rt = line_rt_bnd_bin;
