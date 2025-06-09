function output = f_img_enh(data_mri,opt)

% input: 
%   data_mri: N x M matrix (N image frames; M pixels)
%   opt:      options
% output: 
%   output:   N x M output matrix after image enhancement

% Assign simpler variable names
PCA_thld = opt.enh.PCA_thld;
dr = opt.enh.disk_radius;
mf_win = opt.enh.medfilt_win;
sig_param = opt.enh.sig_param_img;
img_size = opt.img.img_size;

% PCA-based noise reduction
data_mri_nr = f_pca_enh(data_mri, PCA_thld);

output = zeros(size(data_mri));
for t=1:size(data_mri,1)
    
    img = reshape(data_mri_nr(t,:),img_size(1),img_size(2));
    img = f_int_norm(img,1,255);
    
    % compute the pixel sensitivity map
    sen_map = medfilt2(imclose(img,strel('disk',dr)),mf_win);
    sen_map = f_int_norm(sen_map,1,255); % sensitivity map
    img_edge=edge(img,'canny'); % edge of the orignal image 
    sm_img_edge=edge(sen_map,'canny'); % edge of the smoothed image (sensitivity map)
    
    % find the left-most boundary
    % assuming that left-to-right corresponds to the front(face)-to-back direction of the head
    img_lm_bnd = zeros(size(img_edge));
    for i=1:img_size(2)
        bnd_idx1 = find(img_edge(i,:) > 0);  % col1: y (vertical), col2: x (horizontal)
        bnd_idx2 = find(sm_img_edge(i,:) > 0);  % col1: y (vertical), col2: x (horizontal)
        if     (isempty(bnd_idx1) == 0) && (isempty(bnd_idx2) == 0)
            min_x = max([1 min(bnd_idx1) min(bnd_idx2)]);
        elseif (isempty(bnd_idx1) == 0) && (isempty(bnd_idx2) == 1)
            min_x = max([1 min(bnd_idx1)]);
        elseif (isempty(bnd_idx1) == 1) && (isempty(bnd_idx2) == 0)
            min_x = max([1 min(bnd_idx2)]);
        else
            min_x = img_size(2)+1;
        end
        max_x = img_size(2);
        min_y = max([(i-1) 1]);
        max_y = min([(i+1) img_size(2)]);
        img_lm_bnd(min_y:max_y,min_x:max_x) = 1;
    end
    LM_x = zeros(img_size(1),1); % pixel indeces of the left-most boundary of the head
    for i=1:img_size(1)
        bnd_idx = find(img_lm_bnd(i,:) > 0);  % col1: y (vertical), col2: x (horizontal)
        if (isempty(bnd_idx) == 0)
            min_x = max([1 min(bnd_idx)]);
        else
            min_x = img_size(1);
        end
        LM_x(i) = min_x+1;
    end
    % put 1 for the pixels from left-most-edge of the head and neck to the right-most pixel
    % put 0 otherwise
    % this allows to exclude the image background in the sensitivity correction
    LM_bnd = zeros(size(img,1));
    for i=1:img_size(1)
        LM_bnd(i,max([1 min([LM_x(i) img_size(2)])]):end) = 1;
    end
    
    % make background pixel intensity be 0
    img_obj = LM_bnd.*img;
    
    % intensity correction
    img_sen_c = img_obj ./ sen_map;
    img_sen_c = f_int_norm(img_sen_c,1,255);

    % sigmoid warping: highlight tissue and suppress noise
    tmp_img = f_int_norm(img_sen_c,0,1);
    % option for choosing the 2nd parameter of the sigmoid function
    %   in a data-driven way.
    % If you want this option, then uncomment the follwing 3 lines
    % We found that this process does not always useful, so commented it out.
%     [N,X]=hist(reshape(tmp_img,1,img_size(1)*img_size(2)),50);
%     [val,idx]=sort(N,'descend');
%     sig_param(2) = X(idx(2))/2;
    img_w = f_sigmoid_warping(tmp_img,sig_param);
    img_w = f_int_norm(img_w,1,255);
    
    output(t,:) = reshape(img_w,1,img_size(1)*img_size(2));
   
%     % plot
%     h=figure;
%     subplot(331);imagesc((img));title('orig img'); colormap(gray);
%     subplot(332);imagesc((img_edge));title('img edge'); colormap(gray);
%     subplot(333);imagesc((sen_map));title('sm img'); colormap(gray);
%     subplot(334);imagesc((sm_img_edge));title('sm img edge'); colormap(gray);
%     subplot(335);imagesc((LM_bnd));title('LM edge'); colormap(gray);
%     subplot(336);imagesc(img_obj);title('obj img'); colormap(gray);
%     subplot(337);imagesc(img_sen_c);title('sen c img'); colormap(gray); 
%     subplot(338);imagesc(img_w);title('final image'); colormap(gray); 
%     pause(0.05);
%     close(h);
end
