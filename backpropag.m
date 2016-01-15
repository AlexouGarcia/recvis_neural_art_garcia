function [ sol ] = backpropag(net,res,errlayer,grad )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%n=numel(net.layers);
doder=true;
n=errlayer;
dzdy=grad;
cudnn = {'CuDNN'} ;

if doder
  res(n+1).dzdx = dzdy ;
  for i=n:-1:1
    l = net.layers{i} ;
    res(i).backwardTime = tic ;
    switch l.type
      case 'conv'
        %if ~opts.accumulate
          if isfield(l, 'weights')
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                          res(i+1).dzdx, ...
                          'pad', l.pad, 'stride', l.stride, ...
                          cudnn{:}) ;
          else
            % Legacy code: will go
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnconv(res(i).x, l.filters, l.biases, ...
                          res(i+1).dzdx, ...
                          'pad', l.pad, 'stride', l.stride, ...
                          cudnn{:}) ;
          end
%         else
%           dzdw = cell(1,2) ;
%           if isfield(l, 'weights')
%             [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
%                 vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
%                           res(i+1).dzdx, ...
%                           'pad', l.pad, 'stride', l.stride, ...
%                           cudnn{:}) ;
%           else
%             % Legacy code: will go
%             [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
%                 vl_nnconv(res(i).x, l.filters, l.biases, ...
%                           res(i+1).dzdx, ...
%                           'pad', l.pad, 'stride', l.stride, ...
%                           cudnn{:}) ;
%           end
%           for j=1:2
%             res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
%           end
%           clear dzdw ;
       % end

      case 'convt'
        %if ~opts.accumulate
          if isfield(l, 'weights')
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
                          res(i+1).dzdx, ...
                          'crop', l.crop, 'upsample', l.upsample, ...
                          'numGroups', l.numGroups, cudnn{:}) ;
          else
            % Legacy code: will go
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnconvt(res(i).x, l.filters, l.biases, ...
                         res(i+1).dzdx, ...
                          'crop', l.crop, 'upsample', l.upsample, ...
                          'numGroups', l.numGroups, cudnn{:}) ;          end
%         else
%           dzdw = cell(1,2) ;
%           if isfield(l, 'weights')
%             [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
%                 vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
%                           res(i+1).dzdx, ...
%                           'crop', l.crop, 'upsample', l.upsample, ...
%                           'numGroups', l.numGroups, cudnn{:}) ;          else
%             % Legacy code: will go
%             [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
%                 vl_nnconvt(res(i).x, l.filters, l.biases, ...
%                           res(i+1).dzdx, ...
%                           'crop', l.crop, 'upsample', l.upsample, ...
%                           'numGroups', l.numGroups, cudnn{:}) ;          end
%           for j=1:2
%             res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
%           end
%           clear dzdw ;
       % end

      case 'pool'
        res(i).dzdx = vl_nnpool(res(i).x, l.pool, res(i+1).dzdx, ...
                                'pad', l.pad, 'stride', l.stride, ...
                                'method', l.method, ...
                                cudnn{:}) ;
      case 'normalize'
        res(i).dzdx = vl_nnnormalize(res(i).x, l.param, res(i+1).dzdx) ;
      case 'softmax'
        res(i).dzdx = vl_nnsoftmax(res(i).x, res(i+1).dzdx) ;
      case 'loss'
        res(i).dzdx = vl_nnloss(res(i).x, l.class, res(i+1).dzdx) ;
      case 'softmaxloss'
        res(i).dzdx = vl_nnsoftmaxloss(res(i).x, l.class, res(i+1).dzdx) ;
      case 'relu'
        if isfield(l, 'leak'), leak = {'leak', l.leak} ; else leak = {} ; end
        if ~isempty(res(i).x)
          res(i).dzdx = vl_nnrelu(res(i).x, res(i+1).dzdx, leak{:}) ;
        else
          % if res(i).x is empty, it has been optimized away, so we use this
          % hack (which works only for ReLU):
          res(i).dzdx = vl_nnrelu(res(i+1).x, res(i+1).dzdx, leak{:}) ;
        end
      case 'sigmoid'
        res(i).dzdx = vl_nnsigmoid(res(i).x, res(i+1).dzdx) ;
      case 'noffset'
        res(i).dzdx = vl_nnnoffset(res(i).x, l.param, res(i+1).dzdx) ;
      case 'spnorm'
        res(i).dzdx = vl_nnspnorm(res(i).x, l.param, res(i+1).dzdx) ;
      case 'dropout'
        if opts.disableDropout
          res(i).dzdx = res(i+1).dzdx ;
        else
          res(i).dzdx = vl_nndropout(res(i).x, res(i+1).dzdx, ...
                                     'mask', res(i+1).aux) ;
        end
      case 'bnorm'
        if ~opts.accumulate
          if isfield(l, 'weights')
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, ...
                           res(i+1).dzdx) ;
          else
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnbnorm(res(i).x, l.filters, l.biases, ...
                           res(i+1).dzdx) ;
          end
        else
          dzdw = cell(1,2) ;
          if isfield(l, 'weights')
            [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, ...
                           res(i+1).dzdx) ;
          else
            [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                vl_nnbnorm(res(i).x, l.filters, l.biases, ...
                           res(i+1).dzdx) ;
          end
          for j=1:2
            res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
          end
          clear dzdw ;
        end
      case 'pdist'
        res(i).dzdx = vl_nnpdist(res(i).x, l.p, res(i+1).dzdx, ...
                                 'noRoot', l.noRoot, 'epsilon', l.epsilon) ;
      case 'custom'
        res(i) = l.backward(l, res(i), res(i+1)) ;
    end
   % if opts.conserveMemory
   %   res(i+1).dzdx = [] ;
   % end
   % if gpuMode & opts.sync
   %   wait(gpuDevice) ;
   % end
    res(i).backwardTime = toc(res(i).backwardTime) ;
  end
end
sol=res(1).dzdx;

end

