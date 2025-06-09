function f_plot_vtd(data_mri, vt_d, vt_dL, pts_lb, pts_rt, grid, fpath, opt)

% plot estimated tissue-airway boundaries and 
%   corresponding vocal tract distance function.

fr_image = opt.img.fr_image;
img_size = opt.img.img_size;

num_frame = size(data_mri,1);
num_grid = size(grid.bin_pts,1);

switch opt.fig.vtd

  % plot figure
  case 'plot'
    fig=figure('Position', [100 100 400 550]);
    for iFrame = 1:num_frame
      noNaN_vt_dL_idx = find(~isnan(vt_dL(iFrame,:)));
      subplot(7,1,[1:5])
      imagesc(reshape(data_mri(iFrame,:),img_size(1),img_size(2))); colormap(gray); hold on;
      % after smoothing
      plot(pts_lb(iFrame,:,1), pts_lb(iFrame,:,2),'y-','LineWidth',1);
      plot(pts_rt(iFrame,:,1), pts_rt(iFrame,:,2),'y-','LineWidth',1);
      plot(pts_lb(iFrame,noNaN_vt_dL_idx,1), pts_lb(iFrame,noNaN_vt_dL_idx,2),'r-','LineWidth',2);
      plot(pts_rt(iFrame,noNaN_vt_dL_idx,1), pts_rt(iFrame,noNaN_vt_dL_idx,2),'g-','LineWidth',2);
      hold off; title(iFrame);

      subplot(7,1,[6:7])
      plot(vt_d(iFrame,:),'b','LineWidth',3); hold on;
      plot(vt_dL(iFrame,:),'k','LineWidth',3);
      axis([1 num_grid 0 max(max(vt_d))]);
      set(gca,'Xtick',1,'XTickLabel',[]);
      xlabel('[lips]         grid line        [larynx]'); ylabel('mm');
      hold off; pause(1/fr_image);
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
    set(hf, 'position', [150 150 400 550]);
    for iFrame = 1:num_frame
      noNaN_vt_dL_idx = find(~isnan(vt_dL(iFrame,:)));
      subplot(7,1,[1:5])
      imagesc(reshape(data_mri(iFrame,:),img_size(1),img_size(2))); colormap(gray); hold on;
      % after smoothing
      plot(pts_lb(iFrame,:,1), pts_lb(iFrame,:,2),'y-','LineWidth',1);
      plot(pts_rt(iFrame,:,1), pts_rt(iFrame,:,2),'y-','LineWidth',1);
      plot(pts_lb(iFrame,noNaN_vt_dL_idx,1), pts_lb(iFrame,noNaN_vt_dL_idx,2),'r-','LineWidth',2);
      plot(pts_rt(iFrame,noNaN_vt_dL_idx,1), pts_rt(iFrame,noNaN_vt_dL_idx,2),'g-','LineWidth',2);
      hold off; title(iFrame);

      subplot(7,1,[6:7])
      plot(vt_d(iFrame,:),'m','LineWidth',3); hold on;
      plot(vt_dL(iFrame,:),'k','LineWidth',3);
      axis([1 num_grid 0 max(max(vt_d))]);
      set(gca,'Xtick',1,'XTickLabel',[]);
      xlabel('[lips]         grid line        [larynx]'); ylabel('mm');
      hold off; 
      frame = getframe(hf);
      writeVideo(writerObj,frame);
      % writerObj = addframe(writerObj,frame);
    end
    
    close(writerObj);
    close(hf);
    % writerObj=close(writerObj);

  case 'none'
    disp('No plot for estimated cross distance function of the upper airway...');

  otherwise
    Error('check opt.fig.vtd in options.m...');

end
