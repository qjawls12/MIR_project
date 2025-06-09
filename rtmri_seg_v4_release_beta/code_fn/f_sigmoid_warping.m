function y = f_sigmoid_warping(x, param)

% sigmoid function

% If you have any question, please email to jangwon@usc.edu
% Jangwon Kim
% May 2nd 2014


y = 1 ./ (1 + exp((-1) * param(1) * (x - param(2))));
