function output = f_pca_enh(data_mri, PCA_thld)

% PCA image enhancement

% If you have any question, please email to jangwon@usc.edu
% Jangwon Kim
% May 2nd 2014


% principal component analysis
[coeffs, scores, latent] = pca(data_mri);
latent = latent./sum(latent);
cutoff = min(find(cumsum(latent)>PCA_thld));

mean_data_mri = mean(data_mri,1)';
output = zeros(size(data_mri));
for which_frame = 1:size(data_mri,1) %for each frame
    cur_img = zeros(size(data_mri,2),1);
    
    % reconstruct image using weighted components
    for which_comp = 1:cutoff
        cur_comp = coeffs(:,which_comp);
        contrib = scores(which_frame,which_comp);
        cur_img = cur_img + contrib.*cur_comp;
    end
    
    %add the reconstruction to the mean
    cur_img = mean_data_mri+cur_img;
    
    %store the frame
    output(which_frame,:) = cur_img';
end
