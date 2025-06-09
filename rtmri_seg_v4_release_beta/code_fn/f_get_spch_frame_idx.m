function spch_frames = f_get_spch_frame_idx(data_mri,fbname,path,opt)

% inputs:
%   data_mri:	N x M matrix (N image frames; M pixels)
%   fbname:	file basename
%   path:	paths
%   opt:	options
% outputs:
%   spch_frames:	the list of image frames for speech region only
%   frame_idx:		1 for speech frame, 0 for non-speech frame

num_frames = size(data_mri,1);

% If silence label is NOT provided
if isempty(path.sil_lab) == 0
  frame_idx = ones(num_frames,1);
  spch_frames = (1:num_frames)';
% if silence label is provided
else
  [st, et] = textread(fullfile(path.sil_lab, [fbname '.lab_sil']), '%f %f');
  sf = round(st * opt.img.fr_image);
  ef = round(et * opt.img.fr_image);
  frame_idx = ones(num_frames,1);
  for iSil = 1:length(ef)
    cur_sf = max([sf(iSil) 1]);
    cur_ef = min([ef(iSil) num_frames]);
    cur_sil_frames = cur_sf:cur_ef;
    frame_idx(cur_sil_frames) = 0;
  end % iSil
  spch_frames = find(frame_idx);
end

%% Plot MR images and check
%h=figure('Position',[100 100 600 350]);
%noImage = repmat(NaN,opt.img.img_len, opt.img.img_wid);
%for iFrame = 1:num_frames
%  subplot(1,2,1)
%  imagesc(reshape(data_mri(iFrame,:),opt.img.img_len, opt.img.img_wid));
%  title('Original MR images');
%  subplot(1,2,2)
%  if frame_idx(iFrame) == 1
%    imagesc(reshape(data_mri(iFrame,:),opt.img.img_len, opt.img.img_wid));
%  else
%    imagesc(noImage);
%  end
%  title('Only speech-region images');
%  pause(1/opt.img.fr_image);
%end % iFrame
%close(h);
