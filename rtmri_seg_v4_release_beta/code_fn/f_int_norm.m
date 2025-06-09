function output = f_int_norm(img,int_min,int_max)

% image intensity correction
% firstly, limit the intensity range to be in [0.1 - 0.9] quantiles
% then, make intensity range to be in [int_min int_max]
% input:
%   img: M1 x M2 pixel intensity matrix
%   int_min: minimum intensity
%   int_max: maximum intensity
% output:
%   output: M1 x M2 image after changing pixel intensity 

% If you have any question, please email to jangwon@usc.edu
% Jangwon Kim
% May 2nd 2014


[y_len,x_len] = size(img);

if     nargin == 1
    int_min = 0;
    int_max = 255;
elseif nargin == 2
    int_max = 255;
end

% limit intensity to be in [.1 .9] quantiles
img2= reshape(img,1,[]);
q90 = quantile(img2,0.9); 
q90_idx = find(img2 > q90);
q10 = quantile(img2,0.1); 
q10_idx = find(img2 < q10);
img2(q90_idx) = q90;
img2(q10_idx) = q10;

% change the maximum and minimum intensity
img3 = img2 - min(img2);
img4 = ((img3 ./ max(max(img3))) * (int_max - int_min)) + int_min;

output = reshape(img4,y_len,x_len);
