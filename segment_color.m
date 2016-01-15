function [ im_label ] = segment_color( image_in,nb_classes )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% Convert to L*a*b colorspace
cform = makecform('srgb2lab');
lab_he = applycform(image_in,cform);

% kmeans in L*a*b space
ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);

nColors = nb_classes;
% repeat the clustering 3 times to avoid local minima
[cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',3);
% Label all the pixels with the results from kmeans
pixel_labels = reshape(cluster_idx,nrows,ncols);

% Generate images from each label :
segmented_images = cell(1,3);
rgb_label = repmat(pixel_labels,[1 1 3]);

for k = 1:nColors
    color = image_in;
    color(rgb_label ~= k) = 0;
    segmented_images{k} = color;
end

im_label=segmented_images;
end

