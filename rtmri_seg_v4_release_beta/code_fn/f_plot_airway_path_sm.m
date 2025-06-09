function f_plot_airway_path_sm(data_mri, opt_path_sm, opt_path, grid, track_out, opt, fpath)

% Plot/save the estimated airway path

% If you have any question, please email to jangwon@usc.edu
% Jangwon Kim
% May 2nd 2014

% Assign simpler variable name
init_grid_idx_all = track_out.idx_lips;
final_grid_idx_all = track_out.idx_larynx;
fr_image = opt.img.fr_image;
img_size = opt.img.img_size;

num_frame = size(data_mri,1);

% find the pixel coordinate of each airway-path bin
pts_air = f_bin2pts(opt_path, grid, init_grid_idx_all, final_grid_idx_all);
pts_air_sm = f_bin2pts(opt_path_sm, grid, init_grid_idx_all, final_grid_idx_all);

switch opt.fig.airway

  % plot figure
  case 'plot'
    fig=figure('Position', [100 100 400 400]);
    for iFrame = 1:num_frame
      imagesc(reshape(data_mri(iFrame,:,:),img_size(1),img_size(2))); colormap(gray); hold on;
      plot(pts_air(iFrame,:,1),pts_air(iFrame,:,2),'y.');
      plot(pts_air_sm(iFrame,:,1),pts_air_sm(iFrame,:,2),'c.');
      hold off; title(iFrame); pause(0.043);
    end
    close(fig);
  
  % save avi video
  case 'save'
    writerObj = VideoWriter(fpath,'Uncompressed AVI');
    % writerObj = VideoWriter(fpath);
    writerObj.FrameRate = fr_image;
    open(writerObj);
    % writerObj = avifile(fpath,'fps',fr_image);
    hf = figure;
    set(hf, 'position', [150 150 400 400]);
    for iFrame = 1:num_frame
      imagesc(reshape(data_mri(iFrame,:,:),img_size(1),img_size(2))); colormap(gray); hold on;
      plot(pts_air(iFrame,:,1),pts_air(iFrame,:,2),'y.');
      plot(pts_air_sm(iFrame,:,1),pts_air_sm(iFrame,:,2),'c.');
      hold off; title(iFrame); pause(0.043);    
      frame = getframe(hf);
      writeVideo(writerObj,frame);
      % writerObj = addframe(writerObj,frame);
    end

    close(writerObj);
    close(hf);
    % writerObj = close(writerObj);

  case 'none'
    disp('No plot for estimated airway paths...');

  otherwise
    error('check opt.fig.airway in options.m...');

end
