function [img_r,img_g,img_b,img,nrows,ncols] = prepare(img,scale)
img=imresize(img,scale);
img_r=img(:,:,1);
img_g=img(:,:,2);
img_b=img(:,:,3);
[nrows,ncols]=size(img_r);
