function f_plot_lip_larynx(data_mri, grid, track_out, opt, fpath)

% If you have any question, please email to jangwon@usc.edu
% Jangwon Kim
% May 2nd 2014

grid_line_idx_larynx = track_out.idx_larynx;
grid_line_pos_larynx = track_out.pos_larynx;
grid_line_idx_lips = track_out.idx_lips;
grid_line_pos_lips = track_out.pos_lips;
search_width_lips_mm = opt.lip.search_width_lips_mm;
search_length_larynx_mm = opt.lar.search_length_larynx_mm;
fr_image = opt.img.fr_image;
img_size = opt.img.img_size;
px_size = opt.img.px_size;

num_frame = size(data_mri,1);
num_bin = size(grid.bin_pts,2);
num_grid = size(grid.bin_pts,1);

% for lips
search_bin_up = round(search_width_lips_mm(1)/px_size);       % # bins for search (up)
search_bin_down = round(search_width_lips_mm(2)/px_size);     % # bins for search (down)
srch_lip_inner = reshape(grid.bin_pts(grid_line_idx_lips,(round(num_bin/2) + ((-1)*search_bin_down)),:),[],2);
srch_lip_outer = reshape(grid.bin_pts(grid_line_idx_lips,(round(num_bin/2) + search_bin_up),:),[],2);

% for larynx
search_bins = round(search_length_larynx_mm/px_size);
srch_larynx_inner = reshape(grid.bin_pts(grid_line_idx_larynx, (round(num_bin/2) + ((-1)*search_bins)),:),[],2);
srch_larynx_outer = reshape(grid.bin_pts(grid_line_idx_larynx, (round(num_bin/2) + search_bins),:),[],2);

switch opt.fig.lip_larynx

  % plot figure
  case 'plot'
    fig=figure('Position', [100 100 400 400]);
    for iFrame = 1:num_frame
       imagesc(reshape(data_mri(iFrame,:,:),img_size(1),img_size(2))); colormap(gray); hold on;
       line([srch_lip_inner(iFrame,1),srch_lip_outer(iFrame,1)],[srch_lip_inner(iFrame,2),srch_lip_outer(iFrame,2)],'Color',[1 0 0],'LineWidth',2);
       line([srch_larynx_inner(iFrame,1),srch_larynx_outer(iFrame,1)],[srch_larynx_inner(iFrame,2),srch_larynx_outer(iFrame,2)],'Color',[1 1 0],'LineWidth',2);
       hold off; title(iFrame); pause((1/fr_image));
    end
    close(fig);
  
  % save avi video
  case 'save'
    writerObj = VideoWriter(fpath,'Uncompressed AVI');
    %writerObj = VideoWriter(fpath);
    writerObj.FrameRate = fr_image;
    open(writerObj);
    %writerObj = avifile(fpath,'fps',fr_image);
    fig = figure;
    set(fig, 'position', [150 150 400 400]);
    for iFrame = 1:num_frame
       imagesc(reshape(data_mri(iFrame,:,:),img_size(1),img_size(2))); colormap(gray); hold on;
       line([srch_lip_inner(iFrame,1),srch_lip_outer(iFrame,1)],[srch_lip_inner(iFrame,2),srch_lip_outer(iFrame,2)],'Color',[1 0 0],'LineWidth',2);
       line([srch_larynx_inner(iFrame,1),srch_larynx_outer(iFrame,1)],[srch_larynx_inner(iFrame,2),srch_larynx_outer(iFrame,2)],'Color',[1 1 0],'LineWidth',2);
       hold off; title(iFrame); 
       frame = getframe(fig);
       writeVideo(writerObj,frame);
       %writerObj = addframe(writerObj,frame);
    end
    
    close(writerObj);
    close(fig);
    %writerObj = close(writerObj);
  
  case 'none'
    disp('No plot for the results of lip and larynx tracking...');

  otherwise
    error('check opt.fig.lip_larynx in options.m...');

end
