% Project

% Shared variables with fmincon
global net
global rescontent
global resstyle
global resstyle2
global layers_content
global nlayers
global ngrams
global layers_style
global sizelow
global masqueh
global masqueb
global second_style
second_style=1; % set to 1 to mix the style with control, 0 else

%%% Code to launch one time :
vl_setupnn
net = load('imagenet-vgg-verydeep-19.mat');
net2=net;

poolinglayers=[5 10 19 28 ];
for layer=1:length(poolinglayers)
    net2.layers{poolinglayers(layer)}.method='avg';
end
%% Replaced maxpool by average pool
net=net2;


% % Display info :
% vl_simplenn_display(net) ;

%% load images and nets

% 
close all
imcontent=imread('pyramide.jpg');
imstyle=imread('Lichtenstein.jpg');
imstyle2 = imread('Scream.jpg');

sizelow=[250, 250];
%% sizeorig=size(im);

imcontent_ = single(imcontent);
imstyle_   = single(imstyle);
imstyle2_  = single(imstyle2);
imstyle2_  =imresize(imstyle2_,sizelow);
imcontent_ = imresize(imcontent_, sizelow);
imstyle_ = imresize(imstyle_, sizelow);
s=size(imstyle_);
% %imstyle_(floor(s(1)/2):end,:)=imstyle2_(floor(s(1)/2):end,:); % if we
% want to mix the style without control
% 
 im0_=single(rand([sizelow,3])*255);

 resstyle = vl_simplenn(net, imstyle_);
 resstyle2 = vl_simplenn(net, imstyle2_);
 rescontent = vl_simplenn(net, imcontent_);
 res0 = vl_simplenn(net,im0_);
%
% 
% Segmentation - Code from Matlab guide on color clustering
[ im_label ] = segment_color( imcontent,2);
%%% Extract the masks to use in gradient descent controlling the location of style
masqueb=imresize(im_label{1}>0,sizelow);
masqueh=ones([sizelow,3]).*(1-masqueb);
masquenet=masqueh;
%% adding blur on the mask to let the style separation disappear
patch_filt=ones(50,50)/(50*50);
masqueflou=conv2(double(masquenet(:,:,1)),double(patch_filt),'same');
masqueh=masqueflou;
%masqueh=masquenet;
% 'segmented'
%%

layers_content=22;
layers_style=[1,6,11,20,29];
nlayers=length(layers_content);
ngrams=length(layers_style); % nombre de couches utilisées pour la représentation du style

for iter=1:10
      
    options = optimoptions('fmincon','Algorithm','interior-point','GradObj','on','MaxFunEvals',40,'MaxIter',100,'Hessian','lbfgs');
    fun = @energy_grad;
    imin=double(reshape(im0_,[],1)); % feeding a vector in double to fmincon
    x = fmincon(fun,imin,[],[],[],[],0,255,[],options);

    x=reshape(single(x),[sizelow,3]); % reshape the vector to image for display
    im0_=x;
    figure
    imshow(uint8(im0_));
    res0 = vl_simplenn(net,im0_);
end
subplot(2,2,1)
imshow(uint8(im0_));
xlabel('mix content and style')
subplot(2,2,2)
imshow(uint8(imcontent_))
xlabel('content')
subplot(2,2,3)
imshow(uint8(imstyle_))
xlabel('styleh')
subplot(2,2,4)
imshow(uint8(imstyle2_))
xlabel('styleb')
figure
imshow(uint8(im0_));
