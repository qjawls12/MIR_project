function M = Avi2MovieMat_jw(filename)
% Convert avi file to matrix of frames by using mmreader
%
% INPUT: file path
% OUTPUT: the movie matrix (row: frame, col: pixel index)

% If you have any question, please email to jangwon@usc.edu
% Jangwon Kim
% May 2nd 2014


xyloObj = VideoReader(filename);

nFrames = xyloObj.NumberOfFrames;
vidHeight = xyloObj.Height;
vidWidth = xyloObj.Width;

mov(1:nFrames) = ...
    struct('cdata', zeros(vidHeight, vidWidth, 1, 'uint8'),...
           'colormap', []);
       
% Read one frame at a time.
for k = 1 : nFrames
    mov(k).cdata = read(xyloObj, k);
end

vec_length = vidHeight*vidWidth;

M = zeros(nFrames,vec_length);
for itor = 1:nFrames
    M(itor,:) = reshape(double(mov(itor).cdata(:,:,1)),1,vec_length);
end


%eof
