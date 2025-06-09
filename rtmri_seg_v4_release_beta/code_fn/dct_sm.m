function y = dct_sm(x, sm_param, dct_order)

% N dimensional DCT smoothing

num_samp = size(x,1);

lambda = 2 - 2*cos((0:num_samp-1)*pi/num_samp);
gamma = 1./(1+sm_param*lambda.^dct_order)';

y = idct(repmat(gamma,1,size(x,2)).*dct(x));

