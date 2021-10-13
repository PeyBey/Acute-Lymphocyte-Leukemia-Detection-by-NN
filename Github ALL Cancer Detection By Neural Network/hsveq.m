function img_Veq=hsveq(img)
img=im2double(img);
[H,S,V]=rgb2hsv(img);
% Veq=histeq(V);
Veq = imadjust(V);
img_HSVeq=cat(3,H,S,Veq);
img_Veq=hsv2rgb(img_HSVeq);