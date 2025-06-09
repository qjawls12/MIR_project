clear all; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A manual translation (horizontal and vertical)
%   and rotation tool for rtMRI data.
% Manually find the transformation parameters for 
%   one image of each MRI video file to the
%   reference image of a certain MRI video file.
% An MR image (to be transformed) is displayed with
%   the Canny edges of the reference image.
% If you have any question or suggestion, 
%   please email to jangwon@usc.edu
% April 7th 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameters
input_dir = 'rtMRIdata';	% directory for input data
output_dir = 'rtMRIdata_hc';	% directory for output data
ref_img_frame = 30;		% reference image frame
ref_file_idx = 1;		% reference file index
img_size = [68 68];		% image size (# pixels)

% create directory for output data
mkdir(output_dir);

% create the list of input data (mat format)
list_input_files = dir([input_dir '/*.mat']);

% load reference MRI data (mat format)
% variable name: mri_data for this demo file
load([input_dir '/' list_input_files(ref_file_idx).name]);

% obtain the canny edge of the reference image
img_ref = reshape(mri_data(ref_img_frame,:),img_size(1),img_size(2));
img_e = edge(img_ref,'canny'); % edge of the orignal image
[ref_i,ref_j] = ind2sub(size(img_e),find(img_e == 1));

% display the MR image of a certain frame for each MRI video
%   , and decide the transformation parameters
for iFile = 1:length(list_input_files)

  % load MRI video (mat format)
  % variable name: 'mri_data' for this demo file
  load([input_dir '/' list_input_files(iFile).name]);
  mri_data_hc = mri_data;

  % keep repeating the head correction until it is visually well corrected.
  while(1)

    % display the head aligned video
    h=figure('Position', [100 100 500 500]);
    for iFrame = 1:size(mri_data_hc,1)
      imagesc(reshape(mri_data_hc(iFrame,:),img_size(1),img_size(2))); colormap(gray); hold on;
      plot(ref_j,ref_i,'r.'); hold off;
      title(iFrame); pause(0.05);
    end
    close(h);

    % Ask the user for head movement correction
    tmp = input('Do you want the head movement correction? Y(YES) or N(NO) ','s');
    if sum(strcmp(repmat(tmp,4,1),{'Y';'YES';'y';'yes'})) == 1

      % Ask the frame number, image of which is used for manual alignment
      disp(['Total ' num2str(size(mri_data_hc,1)) ' frames']);
      tmp = input('Type the frame number, image of which is used for the correction. (e.g., 30) ','s');
      img_frame_hc = str2num(tmp);

      % choose the image of the frame number   
      img_hc = reshape(mri_data(img_frame_hc,:),img_size(1),img_size(2));

      % manually find the optimal head alignment parameters
      param_opt = f_iter_hc_cor(img_hc,ref_j,ref_i);

      % start image transformation on all frames of the MRI video
      mri_data_hc = zeros(size(mri_data));
      for i=1:size(mri_data,1)
        trans_img = imtransformSimple(reshape(mri_data(i,:),img_size(1),img_size(2)), param_opt);
        mri_data_hc(i,:) = reshape(trans_img,1,img_size(1)*img_size(2));
      end

    else
      % Save the transformed image
      mri_data = mri_data_hc;
      save([output_dir '/' list_input_files(iFile).name],'mri_data');
      disp('(Corrected) final MRI video is saved');
      break;

    end
    
  end

end




