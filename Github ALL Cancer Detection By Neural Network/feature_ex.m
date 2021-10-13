function features_obj=feature_ex(input)

features_obj=zeros(113,size(input,2));
srgb2lab = makecform('srgb2lab');

for i=1:size(input,2)
img_bw=input{1,i};
img_rgb=input{2,i};
img_enh=input{3,i};

%%%%%%%%%%%%%%%%%%%%%%%%% original's sub-bands %%%%%%%%%%%%%%%%%%%%%%%%%%%%
img_rgb_r=img_rgb(:,:,1);
img_rgb_g=img_rgb(:,:,2);
img_rgb_b=img_rgb(:,:,3);
hsv_rgb=rgb2hsv(img_rgb);
img_rgb_h=hsv_rgb(:,:,1);
img_rgb_s=hsv_rgb(:,:,2);
img_rgb_v=hsv_rgb(:,:,3);
lab_rgb=applycform(img_rgb,srgb2lab);
img_rgb_l=lab_rgb(:,:,1);
img_rgb_a=lab_rgb(:,:,2);
img_rgb_B=lab_rgb(:,:,3);

%%%%%%%%%%%%%%%%%%%%%%%%% enhanced's sub-band %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
img_enh_r=img_enh(:,:,1);
img_enh_g=img_enh(:,:,2);
img_enh_b=img_enh(:,:,3);
hsv_enh=rgb2hsv(img_enh);
img_enh_h=hsv_enh(:,:,1);
img_enh_s=hsv_enh(:,:,2);
img_enh_v=hsv_enh(:,:,3);
lab_enh=applycform(img_enh,srgb2lab);
img_enh_l=lab_enh(:,:,1);
img_enh_a=lab_enh(:,:,2);
img_enh_B=lab_enh(:,:,3);

%%%%%%%%%%%%%%%%%%%%% Statistical features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
img_bw=im2bw(img_bw);

mask1=img_rgb_r(img_bw);
mask1=double(mask1);
mask1=normalize01(mask1);
energy1=sum(mask1.*mask1);
ent1=entropy(mask1);
meen1=mean(mask1);
sd1=std(mask1);
kurt1=kurtosis(mask1);
skewn1=skewness(mask1);

mask2=img_rgb_g(img_bw);
mask2=double(mask2);
mask2=normalize01(mask2);
energy2=sum(mask2.*mask2);
ent2=entropy(mask2);
meen2=mean(mask2);
sd2=std(mask2);
kurt2=kurtosis(mask2);
skewn2=skewness(mask2);

mask3=img_rgb_b(img_bw);
mask3=double(mask3);
mask3=normalize01(mask3);
energy3=sum(mask3.*mask3);
ent3=entropy(mask3);
meen3=mean(mask3);
sd3=std(mask3);
kurt3=kurtosis(mask3);
skewn3=skewness(mask3);

mask4=img_rgb_h(img_bw);
mask4=double(mask4);
mask4=normalize01(mask4);
energy4=sum(mask4.*mask4);
ent4=entropy(mask4);
meen4=mean(mask4);
sd4=std(mask4);
kurt4=kurtosis(mask4);
skewn4=skewness(mask4);

mask5=img_rgb_s(img_bw);
mask5=double(mask5);
mask5=normalize01(mask5);
energy5=sum(mask5.*mask5);
ent5=entropy(mask5);
meen5=mean(mask5);
sd5=std(mask5);
kurt5=kurtosis(mask5);
skewn5=skewness(mask5);

mask6=img_rgb_v(img_bw);
mask6=double(mask6);
mask6=normalize01(mask6);
energy6=sum(mask6.*mask6);
ent6=entropy(mask6);
meen6=mean(mask6);
sd6=std(mask6);
kurt6=kurtosis(mask6);
skewn6=skewness(mask6);

mask7=img_rgb_l(img_bw);
mask7=double(mask7);
mask7=normalize01(mask7);
energy7=sum(mask7.*mask7);
ent7=entropy(mask7);
meen7=mean(mask7);
sd7=std(mask7);
kurt7=kurtosis(mask7);
skewn7=skewness(mask7);

mask8=img_rgb_a(img_bw);
mask8=double(mask8);
mask8=normalize01(mask8);
energy8=sum(mask8.*mask8);
ent8=entropy(mask8);
meen8=mean(mask8);
sd8=std(mask8);
kurt8=kurtosis(mask8);
skewn8=skewness(mask8);

mask9=img_rgb_B(img_bw);
mask9=double(mask9);
mask9=normalize01(mask9);
energy9=sum(mask9.*mask9);
ent9=entropy(mask9);
meen9=mean(mask9);
sd9=std(mask9);
kurt9=kurtosis(mask9);
skewn9=skewness(mask9);

mask10=img_enh_r(img_bw);
mask10=double(mask10);
mask10=normalize01(mask10);
energy10=sum(mask10.*mask10);
ent10=entropy(mask10);
meen10=mean(mask10);
sd10=std(mask10);
kurt10=kurtosis(mask10);
skewn10=skewness(mask10);

mask11=img_enh_g(img_bw);
mask11=double(mask11);
mask11=normalize01(mask11);
energy11=sum(mask11.*mask11);
ent11=entropy(mask11);
meen11=mean(mask11);
sd11=std(mask11);
kurt11=kurtosis(mask11);
skewn11=skewness(mask11);

mask12=img_enh_b(img_bw);
mask12=double(mask12);
mask12=normalize01(mask12);
energy12=sum(mask12.*mask12);
ent12=entropy(mask12);
meen12=mean(mask12);
sd12=std(mask12);
kurt12=kurtosis(mask12);
skewn12=skewness(mask12);

mask13=img_enh_h(img_bw);
mask13=double(mask13);
mask13=normalize01(mask13);
energy13=sum(mask13.*mask13);
ent13=entropy(mask13);
meen13=mean(mask13);
sd13=std(mask13);
kurt13=kurtosis(mask13);
skewn13=skewness(mask13);

mask14=img_enh_s(img_bw);
mask14=double(mask14);
mask14=normalize01(mask14);
energy14=sum(mask14.*mask14);
ent14=entropy(mask14);
meen14=mean(mask14);
sd14=std(mask14);
kurt14=kurtosis(mask14);
skewn14=skewness(mask14);

mask15=img_enh_v(img_bw);
mask15=double(mask15);
mask15=normalize01(mask15);
energy15=sum(mask15.*mask15);
ent15=entropy(mask15);
meen15=mean(mask15);
sd15=std(mask15);
kurt15=kurtosis(mask15);
skewn15=skewness(mask15);

mask16=img_enh_l(img_bw);
mask16=double(mask16);
mask16=normalize01(mask16);
energy16=sum(mask16.*mask16);
ent16=entropy(mask16);
meen16=mean(mask16);
sd16=std(mask16);
kurt16=kurtosis(mask16);
skewn16=skewness(mask16);

mask17=img_enh_a(img_bw);
mask17=double(mask17);
mask17=normalize01(mask17);
energy17=sum(mask17.*mask17);
ent17=entropy(mask17);
meen17=mean(mask17);
sd17=std(mask17);
kurt17=kurtosis(mask17);
skewn17=skewness(mask17);

mask18=img_enh_B(img_bw);
mask18=double(mask18);
mask18=normalize01(mask18);
energy18=sum(mask18.*mask18);
ent18=entropy(mask18);
meen18=mean(mask18);
sd18=std(mask18);
kurt18=kurtosis(mask18);
skewn18=skewness(mask18);

%%%%%%%%%%%%%%%%%%%%%%%%% Geometrical features %%%%%%%%%%%%%%%%%%%%%%%%%%%%
s  = regionprops(img_bw, 'Area','Eccentricity','Solidity','Extent','Perimeter');
area=[s.Area];
ecc=[s.Eccentricity];
sol=[s.Solidity];
ext=[s.Extent];
per=[s.Perimeter];

%%%%%%%%%%%%%%%%%%%%%%%%% data train %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
features_obj(1,i)=energy1;features_obj(2,i)=ent1;features_obj(3,i)=meen1;
features_obj(4,i)=sd1;features_obj(5,i)=skewn1;features_obj(6,i)=kurt1;
features_obj(7,i)=energy2;features_obj(8,i)=ent2;features_obj(9,i)=meen2;
features_obj(10,i)=sd2;features_obj(11,i)=skewn2;features_obj(12,i)=kurt2;
features_obj(13,i)=energy3;features_obj(14,i)=ent3;features_obj(15,i)=meen3;
features_obj(16,i)=sd3;features_obj(17,i)=skewn3;features_obj(18,i)=kurt3;
features_obj(19,i)=energy4;features_obj(20,i)=ent4;features_obj(21,i)=meen4;
features_obj(22,i)=sd4;features_obj(23,i)=skewn4;features_obj(24,i)=kurt4;
features_obj(25,i)=energy5;features_obj(26,i)=ent5;features_obj(27,i)=meen5;
features_obj(28,i)=sd5;features_obj(29,i)=skewn5;features_obj(30,i)=kurt5;
features_obj(31,i)=energy6;features_obj(32,i)=ent6;features_obj(33,i)=meen6;
features_obj(34,i)=sd6;features_obj(35,i)=skewn6;features_obj(36,i)=kurt6;
features_obj(37,i)=energy7;features_obj(38,i)=ent7;features_obj(39,i)=meen7;
features_obj(40,i)=sd7;features_obj(41,i)=skewn7;features_obj(42,i)=kurt7;
features_obj(43,i)=energy8;features_obj(44,i)=ent8;features_obj(45,i)=meen8;
features_obj(46,i)=sd8;features_obj(47,i)=skewn8;features_obj(48,i)=kurt8;
features_obj(49,i)=energy9;features_obj(50,i)=ent9;features_obj(51,i)=meen9;
features_obj(52,i)=sd9;features_obj(53,i)=skewn9;features_obj(54,i)=kurt9;
features_obj(55,i)=energy10;features_obj(56,i)=ent10;features_obj(57,i)=meen10;
features_obj(58,i)=sd10;features_obj(59,i)=skewn10;features_obj(60,i)=kurt10;
features_obj(61,i)=energy11;features_obj(62,i)=ent11;features_obj(63,i)=meen11;
features_obj(64,i)=sd11;features_obj(65,i)=skewn11;features_obj(66,i)=kurt11;
features_obj(67,i)=energy12;features_obj(68,i)=ent12;features_obj(69,i)=meen12;
features_obj(70,i)=sd12;features_obj(71,i)=skewn12;features_obj(72,i)=kurt12;
features_obj(73,i)=energy13;features_obj(74,i)=ent13;features_obj(75,i)=meen13;
features_obj(76,i)=sd13;features_obj(77,i)=skewn13;features_obj(78,i)=kurt13;
features_obj(79,i)=energy14;features_obj(80,i)=ent14;features_obj(81,i)=meen14;
features_obj(82,i)=sd14;features_obj(83,i)=skewn14;features_obj(84,i)=kurt14;
features_obj(85,i)=energy15;features_obj(86,i)=ent15;features_obj(87,i)=meen15;
features_obj(88,i)=sd15;features_obj(89,i)=skewn15;features_obj(90,i)=kurt15;
features_obj(91,i)=energy16;features_obj(92,i)=ent16;features_obj(93,i)=meen16;
features_obj(94,i)=sd16;features_obj(95,i)=skewn16;features_obj(96,i)=kurt16;
features_obj(97,i)=energy17;features_obj(98,i)=ent17;features_obj(99,i)=meen17;
features_obj(100,i)=sd17;features_obj(101,i)=skewn17;features_obj(102,i)=kurt17;
features_obj(103,i)=energy18;features_obj(104,i)=ent18;features_obj(105,i)=meen18;
features_obj(106,i)=sd18;features_obj(107,i)=skewn18;features_obj(108,i)=kurt18;
features_obj(109,i)=area;features_obj(110,i)=per;features_obj(111,i)=sol;
features_obj(112,i)=ext;features_obj(113,i)=ecc;
end