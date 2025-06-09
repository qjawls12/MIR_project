function y = f_norm_min_max(x, min_y, max_y)

% normalize the pixel intensity of each image so that
%   the range of the pixel intenxity is [min_y max_y]

y = (((x - min(min(x))) ./ (max(max(x)) - min(min(x)))) .* (max_y - min_y)) + min_y;
