function [Iout] = preprocess_image_malaria(filename)
% This function preprocess the malaria images by reshaping and histogram
% equalization
%
I = imread(filename);
I = imresize(I,[227 227]);
% I = rgb2gray(I);
r = I(:,:,1);
mean_r = mean(r(:));
g = I(:,:,2);
mean_g = mean(g(:));
b = I(:,:,1);
mean_b = mean(b(:));
ov = mean(I(:));
I(:,:,1) = double(I(:,:,1))*(ov/mean_r);
I(:,:,2) = double(I(:,:,2))*(ov/mean_g);
I(:,:,3) = double(I(:,:,3))*(ov/mean_b);
Iout = uint8(I);
% Iout = histeq(I);


end

