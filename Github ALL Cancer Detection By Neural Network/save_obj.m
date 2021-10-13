function objects=save_obj(bw,img,img_Veq)


[bw_label,n]=bwlabel(bw);
for i=1:n
    L=bw_label;
    L(L~=i)=0;
    L(L==i)=1;
    data_BB= regionprops(L, 'BoundingBox');
    BB=data_BB.BoundingBox;
    I = imcrop(L, BB);
    obj_bw{1,i}=I;
end

for i=1:n
    L=bw_label;
    L(L~=i)=0;
    L(L==i)=1;
    data_BB= regionprops(L, 'BoundingBox');
    BB=data_BB.BoundingBox;
    BB_r=uint8(immultiply(double(L),double(img(:,:,1))));
    BB_g=uint8(immultiply(double(L),double(img(:,:,2))));
    BB_b=uint8(immultiply(double(L),double(img(:,:,3))));
    nuc_rgb=cat(3,BB_r,BB_g,BB_b);
    I = imcrop(nuc_rgb, BB);
    obj_rgb{1,i}=I;
end

img_Veq=im2uint8(img_Veq);
for i=1:n
    L=bw_label;
    L(L~=i)=0;
    L(L==i)=1;
    data_BB= regionprops(L, 'BoundingBox');
    BB=data_BB.BoundingBox;
    BB_r=uint8(immultiply(double(L),double(img_Veq(:,:,1))));
    BB_g=uint8(immultiply(double(L),double(img_Veq(:,:,2))));
    BB_b=uint8(immultiply(double(L),double(img_Veq(:,:,3))));
    nuc_rgb=cat(3,BB_r,BB_g,BB_b);
    I = imcrop(nuc_rgb, BB);
    obj_enh{1,i}=I;
end



for i=1:n
objects{1,i}=obj_bw{1,i};
objects{2,i}=obj_rgb{1,i};
objects{3,i}=obj_enh{1,i};
end