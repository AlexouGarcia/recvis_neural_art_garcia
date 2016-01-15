function [energyglobal,gradglobal] = energy_grad(imin)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
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

alpha=1;
ratio=10^-1;

beta=alpha/ratio;
somme=alpha+beta;
alpha=alpha/somme;
beta=beta/somme;


    im0_=reshape(single(imin),[sizelow,3]);
    
    firstiter=0;
    res0 = vl_simplenn(net,im0_);
    for layers=1:nlayers
       
        % Content :
        [energy, gradcontx] = rawgradient_content_x( net,rescontent,res0,layers_content,layers );
       if firstiter==0
           firstiter=1;
           gradglobal=alpha*gradcontx/nlayers;
           energyglobal=alpha*energy/nlayers;           
       else
           gradglobal=gradglobal+alpha*gradcontx/nlayers;
           energyglobal=energyglobal+alpha*energy/nlayers;
       end
    end
        % Style :
        
    for layers=1:ngrams
        if second_style==1
            [ gradstylex,energystyle ] = gradient_style_duoim_x( net,resstyle,...
    resstyle2,res0,layers_style,layers,masqueh );
        else
            [energystyle, gradstylex ] = rawgradient_style_x( net,resstyle,res0,layers_style,layers );
        end
        
        if firstiter==0
            firstiter=1;
            gradglobal=beta*gradstylex/nlayers;
            energyglobal=beta*energystyle/nlayers;
            
        else
            gradglobal=gradglobal+beta*gradstylex/nlayers;
            energyglobal=energyglobal+beta*energystyle/nlayers; 
            
        end
    end   
        %%% smoothness constraint %%%
        energy_tikhonov=1/2*sum(sum(sum((im0_-circshift(im0_,[1,0,0])).^2+(im0_-circshift(im0_,[0,1,0]))).^2));
        grad_tikhonov=im0_-circshift(im0_,[-1,0,0])+(im0_-circshift(im0_,[0,-1,0]));
        energyglobal=energyglobal+10^-1*energy_tikhonov;
        gradglobal=gradglobal+10^-1*grad_tikhonov;
        
        
        
    gradglobal=double(reshape(gradglobal,[],1));
    energyglobal=double(energyglobal);
    'step'
    

end

