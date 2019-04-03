%%   Unsupervised Change Detection using PCA and K-Means Function
%   _______________________________________________________________________
%   Le Duc Khai
%   Bachelor in Biomedical Engineering
%   FH Aachen - University of Applied Sciences, Germany.
%
%   Released on 01.04.2019.
%
%   The proposed algorithm detects changes between 2 satellite images using
%   Principle Component Analysis (PCA) and K-Means Clustering.
%
%   Implementation is based on this scientific paper:
%       Turgay Celik
%       "Unsupervised Change Detection in Satellite Images Using Principal Component Analysis
%       and k-Means Clustering"
%       IEEE GEOSCIENCE AND REMOTE SENSING LETTERS, VOL. 6, NO. 4, OCTOBER 2009 
%       DOI: 10.1109/LGRS.2009.2025059
%
%   The following codes are implemented only for PERSONAL USE, e.g improving
%   programming skills in the domain of Image Processing and Computer Vision.
%   If you use this algorithm, please cite the paper mentioned above to support
%   the authors.
%
%   Parameters:
%       image1: the 1st input image, should be in RGB-scale
%       image2: the 2nd input image, should be in RGB-scale
%       h: the block size, h is an integer
%       rate: coefficient of thresholding
%
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
function [change_map] = ChangeDetection_PCA_KMeans(image1, image2, h, rate)
%% Set input parameters
if size(image1) ~= size(image2)
    warning('Image 1 and 2 must be in the same size');
end
[rows cols ~] = size(image1);
if ndims(image1) ~= 3
    image1 = repmat(image1, [1 1 3])/255;
    image2 = repmat(image2, [1 1 3])/255;
end
figure(1);
subplot(2,2,1); imshow(image1); title('Original image 1');
subplot(2,2,2); imshow(image2); title('Original image 2');

%% Calculate difference image (Equation 1)
dif_image = abs(double(image1 - image2));
subplot(2,2,3); imshow(dif_image,[]); title('Difference image');

%% Padding for the difference image
pad = zeros(rows+h-1, cols+h-1, 3);
pad(1:rows, 1:cols, :) = dif_image;

%% Divide the difference image into h x h non-overlapping blocks (Equation 2)
vector_set = zeros(rows*cols, 3*h*h);
count = 1;
for i = 1:rows
    for j = 1:cols
        block = pad(i:i+h-1, j:j+h-1, :);
        vector_set(count, :) = reshape(block, 1, []);
        count = count + 1;
    end
end
clear count;

%% PCA algorithm
% Calculate average vector of the set (Equation 3)
mean_vector = mean(vector_set,1);
dif_vector = vector_set - mean_vector;
% Calculate covariance matrix (Equation 4)
covar = cov(dif_vector);
% Compute eigenvectors and eigenvalues of the covariance matrix
[eigvector eigvalue_mat] = eig(covar);
eigvalue = diag(eigvalue_mat);
% Sort the eigen vectors according to the descending eigenvalues
[~, index] = sort(-eigvalue);
eigvalue = eigvalue(index);
eigvector = abs(eigvector(:,index));
clear junk;
% Choose number of eigenvectors satisfied the required magnitude percentage
EigenvectorPer = 1;
lim = ceil(EigenvectorPer*size(eigvector,2));
eigvector = eigvector(:, 1:lim);
% Choose number of eigenvectors satisfied the thresholding
for k1 = length(eigvalue):-1:1
    if(sum(eigvalue(k1:length(eigvalue)))>=rate*sum(eigvalue))
        break;
    end
end
eigvector = eigvector(:,k1:length(eigvalue));
% Project difference image vector set onto eigenvector space to get feature vector space
feature = vector_set * eigvector;

%% K-Means algorithm
[label,~] = kmeans(feature,2);
change_map = reshape(label, [rows cols])';
change_map = change_map - 1;
subplot(2,2,4); imshow(change_map, []); title('Change map');

end

