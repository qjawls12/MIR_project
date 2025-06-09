function B = quanfilt3(A,siz,Q)

% This performs 3D quanfile smoothing for 3D matrix
% This code is slighly modified from 'MEDFILT3.m' by Jangwon Kim
%   for performing 3D quantile filtering.
%   MEDFILT3.m was originally written by Damien Garcia.
%   his website: <a
%   href="matlab:web('http://www.biomecardio.com')">www.BiomeCardio.com</a>
%
% Please email to Jangwon Kim for any question and suggestion.
% jangwon@usc.edu

sizA = size(A);

%% Chunks: the numerical process is split up in order to avoid large arrays
N = numel(A);
siz = ceil((siz-1)/2);
n = prod(siz*2+1);
nchunk = (1:ceil(N/n):N);
if nchunk(end)~=N, nchunk = [nchunk N]; end

%% Change to double if needed
class0 = class(A);
if ~isa(A,'float')
    A = double(A);
end

%% Padding neighbors on the edges 
B = A;
sizB = sizA;
A = padarray(A,siz,'replicate');
sizA = size(A);

%% Creating the index arrays (INT32)
inc = zeros([3 2*siz+1],'int32');
siz = int32(siz);
[inc(1,:,:,:) inc(2,:,:,:) inc(3,:,:,:)] = ndgrid(...
    [0:-1:-siz(1) 1:siz(1)],...
    [0:-1:-siz(2) 1:siz(2)],...
    [0:-1:-siz(3) 1:siz(3)]);
inc = reshape(inc,1,3,[]);

I = zeros([sizB 3],'int32');
sizB = int32(sizB);
[I(:,:,:,1) I(:,:,:,2) I(:,:,:,3)] = ndgrid(...
    (1:sizB(1))+siz(1),...
    (1:sizB(2))+siz(2),...
    (1:sizB(3))+siz(3));
I = reshape(I,[],3);

%% Filtering
for i = 1:(length(nchunk)-1)

    Im = repmat(I(nchunk(i):nchunk(i+1),:),[1 1 n]);
    Im = bsxfun(@plus,Im,inc);

    I0 = Im(:,1,:) +...
        (Im(:,2,:)-1)*sizA(1) +...
        (Im(:,3,:)-1)*sizA(1)*sizA(2);
    I0 = squeeze(I0);
    B(nchunk(i):nchunk(i+1)) = quantile(A(I0),Q,2);
end
B = cast(B,class0);
    
