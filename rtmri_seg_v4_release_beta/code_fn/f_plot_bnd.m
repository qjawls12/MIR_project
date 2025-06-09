function f_plot_bnd(data_mri, grid, track_out, bnd_out, opt, fpath)

% plot (i) only interpolated tissue-airway boundary points
%      (ii) smoothed tissue-airway boundary points

% If you have any question, please email to jangwon@usc.edu
% Jangwon Kim
% Dec 18 2014

% Assign simpler variable name
bin_lb = bnd_out.bin_lb_sm;
bin_rt = bnd_out.bin_rt_sm;
bin_lb_i = bnd_out.bin_lb_i;
bin_rt_i = bnd_out.bin_rt_i;
init_grid_idx_all = track_out.idx_lips;
final_grid_idx_all = track_out.idx_larynx;
fr_image = opt.img.fr_image;
img_size = opt.img.img_size;

num_frame = size(data_mri,1);

% find corresponding point for each boundary point
pts_lb_sm = f_bin2pts(bin_lb, grid, init_grid_idx_all, final_grid_idx_all);
pts_rt_sm = f_bin2pts(bin_rt, grid, init_grid_idx_all, final_grid_idx_all);    
pts_lb_i = f_bin2pts(bin_lb_i, grid, init_grid_idx_all, final_grid_idx_all);
pts_rt_i = f_bin2pts(bin_rt_i, grid, init_grid_idx_all, final_grid_idx_all); 


switch opt.fig.bnd

  % plot figure
  case 'plot'
    fig=figure('Position', [100 100 400 400]);
    for iFrame = 1:num_frame
      imagesc(reshape(data_mri(iFrame,:),img_size(1),img_size(2))); colormap(gray); hold on;
      plot(pts_lb_i(iFrame,:,1),pts_lb_i(iFrame,:,2),'y.');
      plot(pts_rt_i(iFrame,:,1),pts_rt_i(iFrame,:,2),'y.');
      plot(pts_lb_sm(iFrame,:,1),pts_lb_sm(iFrame,:,2),'r-','LineWidth',3);
      plot(pts_rt_sm(iFrame,:,1),pts_rt_sm(iFrame,:,2),'g-','LineWidth',3);
      scatter(grid.mpal(1),grid.mpal(2),'m.');
      hold off; title(iFrame); pause(0.043);
    end
    close(fig);
  
  % save avi video
  case 'save'
    writerObj = VideoWriter(fpath,'Uncompressed AVI');
    %writerObj = VideoWriter(fpath);
    writerObj.FrameRate = fr_image;
    open(writerObj);
    % writerObj = avifile(fpath,'fps',fr_image);
    hf = figure;
    set(hf, 'position', [150 150 400 400]);
    for iFrame = 1:num_frame
      imagesc(reshape(data_mri(iFrame,:),img_size(1),img_size(2))); colormap(gray); hold on;
      plot(pts_lb_i(iFrame,:,1),pts_lb_i(iFrame,:,2),'y.');
      plot(pts_rt_i(iFrame,:,1),pts_rt_i(iFrame,:,2),'y.');
      plot(pts_lb_sm(iFrame,:,1),pts_lb_sm(iFrame,:,2),'r-','LineWidth',3);
      plot(pts_rt_sm(iFrame,:,1),pts_rt_sm(iFrame,:,2),'g-','LineWidth',3);
      scatter(grid.mpal(1),grid.mpal(2),'m.');
      hold off; title(iFrame);

      frame = getframe(hf);
      writeVideo(writerObj,frame);
      % writerObj = addframe(writerObj,frame);
    end
    
    close(writerObj);
    close(hf);
    % writerObj=close(writerObj);

  case 'none'
    disp('No plot for estimated tissue-airway boundaries...');

  otherwise
    Error('check opt.fig.bnd in options.m...');

end
