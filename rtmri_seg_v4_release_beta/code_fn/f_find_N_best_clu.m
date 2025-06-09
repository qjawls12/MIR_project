function g_idx_b = f_find_N_best_clu(data,Nclu,NMaxClu)

% Find the samples of the N best clusters.
% Agglomerative hierarchical clustering is performed
%   based on the distance of each sample to the median
% inputs:
%    data: Nx1 verctor
%    Nclu: the number of clusters of frames
%    NMaxClu: the maximum number of clusters (recommended: 2 or 3)
%
% If you have any question, please email to jangwon@usc.edu
% Jangwon Kim
% Dec 18 2014


% agglomerative hierarchical clustering of airway path bins
%data_m = repmat(median(data),size(data));
%Z = linkage([data data_m],'single','euclidean');
Z = linkage(data,'single','euclidean');
T = cluster(Z,'maxclust',NMaxClu);
num_link = length(T);
clu_list=unique(sort(T));
num_clu = length(clu_list);
cnt_pts_clu = zeros(num_clu,1);
for which_clu = 1:num_clu
    cnt_pts_clu(which_clu) = sum(T == repmat(which_clu,length(T),1));
end
[cnt_pts_clu_sort,idx] = sort(cnt_pts_clu,'descend');
N_best_clu = clu_list(idx(1:Nclu));
g_idx_b = zeros(length(data),1);
for which_clu = 1:length(N_best_clu)
  g_idx_b(find(T == N_best_clu(which_clu))) = 1;
end

