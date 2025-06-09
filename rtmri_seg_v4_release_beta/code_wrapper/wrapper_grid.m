

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is a wrapper for constructing grid lines
%   and save them.
% If you have any question, please email to
%   jangwon@usc.edu
% Dec 18 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../code_fn');
addpath('../config');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PARAMETERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options;
paths;

% create a directory for output
system(['mkdir -p ' path.output_data_dir]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GRID LINE CONSTRUCTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Manually choose
%   (i) the center line of the grid lines
%   (ii) landmark points
grid = f_grid_manual(opt, path);

% Construct grid lines
grid = f_const_gridline(grid, opt);

% Plot and check all grid lines
f_plot_grid_manual(grid, opt, path);

% Save the grid line
fpath = fullfile(path.output_data_dir, 'grid.mat');
save(fpath,'grid');

