% MR Image parameters
opt.img.fr_image = 23.180;		% rtMRI video frame rate
opt.img.px_size = 2.9;			% pixel length (width) = 2.9mm
opt.img.img_len = 68;			% length (# pixels) of one MR image
opt.img.img_wid = 68;			% width (# pixels) of one MR image
opt.img.img_size = [opt.img.img_len, opt.img.img_wid];	% the size of image

% parameters for grid lines
opt.grid.gwidth  = 23;			% initial width of Ohman analysis grid (px)
opt.grid.gint_mm = 2;			% intervals between centers of adjacent gridlines (mm)
opt.grid.ref_img_idx = 40;		% image frame number to be used for choosing landmark points
opt.grid.mpal_mrg = 3;			% margin of the center of the highest palatal grid line
					%   from the palatal landmark point
opt.grid.grid_lip_prot_mm = 20;		% length (mm) of maximum lip protrusion
					%   from the labial landmark point
opt.grid.grid_larynx_motion_mm = 20;	% range (mm) of larynx lowering 
					%   from the larynx landmark point
opt.grid.num_grid_lip_prot = opt.grid.grid_lip_prot_mm / opt.grid.gint_mm; % number of grid lines for
					%   lip protrusion from the labial landmark point
opt.grid.num_grid_larynx_motion = opt.grid.grid_larynx_motion_mm / opt.grid.gint_mm; % number of grid lines for
					%   larynx lowering from the labial landmark point
opt.grid.gint    = opt.grid.gint_mm / opt.img.px_size;	% intervals between gridlines (px)
opt.grid.num_bin = opt.grid.gwidth*2;		% number of bins

% DCT smoothing parameter for centers of grid lines
opt.grid.dct_order = 3;			% order of smoothing process: smaller, smoother (default: 3)
opt.grid.sm_param = 2*10^2;		% smoothing parameter: greater, smoother (default: 10^2)

% parameters for lip and larynx detection
opt.lip.search_length_lips_mm = 25;	% search window length (X mm forward and X mm backward)
opt.lip.search_width_lips_mm = [30 20];	% search window width (X(1) mm upward and X(2) mm downward)
opt.lar.search_length_larynx_mm = 15;	% search window length (X mm forward and X mm backward)
opt.lar.search_width_larynx_mm = 20;	% search window width (X mm upward and X mm downward)
opt.lar.sig_param_larynx = [20 0.3];	% sigmoid warping function parameters for 
					%   transition cost of larynx estimation

% image enhancement parameters
% (0) PCA-based noise reduction
opt.enh.PCA_thld = 0.90;                % for PCA noise reduction (how much cumul. eig. values)
% (1) pixel sensitivity correction
opt.enh.disk_radius = 10;			% morphological 
opt.enh.medfilt_win = [5 5];			% [M N]: M-by-N neighborhood around the corresponding pixel
% (2) sigmoid warping
% col 1: higher => stiffer; range: [0 inf]
% col 2: higher => higher center of the function; range: [0 1]
opt.enh.sig_param_img=[40 0.5];

% parameters for airway-path estimation
opt.airway.obj_sm = 1;				% optional objective cost matrix smoothing (default = 1)
					%   1: smoothing by imposing the mean of the quantiles of the neighboring data
					%   0: no smoothing
opt.airway.sigmoid_param_air = [20 0.25];	% sigmoid warping function parameters for 
						%   transition cost of airway-path estimation
opt.airway.sigmoid_param_air_lip = [30 0.05];	% sigmoid warping function parameters for
						%   transition cost of airway-path estimation in the
						%     region from the alveolar ridge landmark point
						%     to the front-most edge of the lips
opt.airway.trans_w = 1;			% weight on smoothness of airway path curve (higher, smoother)
opt.airway.obslik_sm_frame_span = 3;	% objective cost smoothing using (2X+1) frames
opt.airway.obslik_sm_grid_span = 3;	% objective cost smoothing using (2Y+1) grid lines
opt.airway.obslik_sm_bin_span = 3;	% objective cost smoothing using (2Z+1) bins
opt.airway.outer_dist = 2;		% maximum range (pixels) for outer boundary from the center points of the grid lines
					%   if you don't want this constraint, then set it to [].
opt.airway.inner_dist = 3;		% maximum range (pixels) for inner boundary from the center points of the grid lines
opt.airway.num_best_clu = 1;		% the number of clusters of frames for airway-path smoothing
opt.airway.num_max_clu = 3;		% the maximum number of clusters (recommended: 2 or 3) for airway-path smoothing


% parameter for tissue-airway boundary estimation and smoothing
opt.bnd.bnd_thld = 0.5;			% threshold where the boundary is determined
opt.bnd.win_span_sm = 6;		% span X pixels from the center to each side (left,right)

% DCT smoothing parameters for boundaries
opt.bnd.dct_order = 4;		% order of smoothing process: smaller, smoother (default: 4)
opt.bnd.sm_param = 10;		% smoothing parameter: greater, smoother (default: 10)
opt.bnd.phar.num_best_clu = 1;	% the number of clusters of frames for pharyngeal wall boundary
opt.bnd.phar.num_max_clu = 2;	% the maximum number of clusters for pharyngeal wall boundary
opt.bnd.phar.sm_param = 10^2;	
opt.bnd.phar.dct_order = 2;	

% Options ofr figures
% 'save' to save movie; 
% 'plot' to just display movie without saving
% 'none' to neither save nor display movie
opt.fig.lip_larynx = 'plot';	% lip and larynx tracking
opt.fig.airway = 'plot';	% airway path
opt.fig.bnd = 'plot';		% tissue-airway boundaries
opt.fig.vtd = 'plot';		% vocal tract cross distance function
