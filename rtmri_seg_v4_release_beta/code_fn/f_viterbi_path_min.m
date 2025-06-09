function best_state_path = f_viterbi_path_min(prior, cost_mat, transmat)

% This code finds the path of the minimum cost through HMM trellis, using 
%   Viterbi algorithm
% Input:
%   Assuming that N = # instances, M = # states,
%   prior: 1 x M prior score vector
%   cost_mat: N x M cost matrix
%   transmat: (N-1) x 1 cell vector. each cell has M x M matrix
% Output:
%   best_state_path: N x 1 best state sequence
% 
% This code is written by Jangwon Kim
%   Signal Analysis and Interpretation Lab.
%   University of Southern California, Los Angeles, USA
% Email to jangwon@usc.edu for any question and suggestion

num_instance = size(transmat,1)+1;

% compute the sum of scores through state trellis
best_prev_idx = cell(num_instance-1,1);
upto_cur_score_all = cell(num_instance,1);
upto_cur_score_all{1} = cost_mat(1,:) .* prior; % for 1st phn

for which_inst=2:num_instance
    cur_state_score = cost_mat(which_inst,:);
    best_prev_idx{which_inst-1} = zeros(length(cur_state_score),1);
    min_transed_score = zeros(1,length(cur_state_score));
    for which_cur_state = 1:length(cur_state_score)
        [min_transed_score(which_cur_state), best_prev_idx{which_inst-1}(which_cur_state)] = ...
            min(upto_cur_score_all{which_inst-1}' + transmat{which_inst-1}(:,which_cur_state));
    end
    upto_cur_score_all{which_inst} = min_transed_score + cur_state_score;
end

% find the lowest score path recursively
best_state_path = zeros(num_instance,1);
[dum_minval,best_state_path(num_instance)] = min(upto_cur_score_all{num_instance});
for which_inst=(num_instance-1):-1:1
    best_state_path(which_inst) = best_prev_idx{which_inst}(best_state_path(which_inst+1));
end
