function NN=Initialization(LayerStruct,NN)
    
    if nargin==1
            NN=struct;
    end

    if isfield(NN,'Default')==1
        clear NN;
        NN.ActivationFunction='Gaussian';
        NN.Cost='MSE';
        NN.NetworkType='ANN';
        NN.InputAutoScaling='on';
        NN.LabelAutoScaling='off';
    end

    if isfield(NN,'ActivationFunction')~=1
        NN.ActivationFunction='Gaussian';
    end
    
    if isfield(NN,'Cost')~=1
        NN.Cost='MSE';
    end
    
    NN.MeanFactor=1;


    if isfield(NN,'InputAutoScaling')~=1
        NN.InputAutoScaling='off';
    end

    if isfield(NN,'LabelAutoScaling')~=1
        NN.LabelAutoScaling='off';
    end

    LayerMatrix(1,:)=LayerStruct;
    LayerMatrix(2,1:end-1)=LayerStruct(2:end);
    NumOfLayer=length(LayerMatrix(1,:))-1;
    NN.depth=NumOfLayer;
    NN.numOfParameters=LayerMatrix(1,:)*LayerMatrix(2,:)'+sum(LayerMatrix(2,:));
    NN.numOfWeight=LayerMatrix(1,:)*LayerMatrix(2,:)';
    NN.numOfBias=sum(LayerMatrix(2,:));
    NN.LayerStruct=LayerMatrix;
    
    
    if isfield(NN,'NetworkType')~=1
        NN.NetworkType='ANN';
    end
    ResidualOn=strcmp(NN.NetworkType,'ResNet');
    if isfield(NN,'LineSearcher')==0
        NN.LineSearcher='BackTrack';
    end
    
    NN.PreTrained=0;

    
    if strcmp(NN.Cost,'Entropy')==1
        NN.OutActive=@(x) SoftMax(x);
        NN.IndexToHot=@(x) IndexToHot(x);
        NN.HotToIndex=@(x) HotToIndex(x);
    else
        NN.OutActive=@(x) x;
    end


    activation=NN.ActivationFunction;
    if isa(NN.ActivationFunction,'function_handle')==0
        switch activation
            case 'Gaussian'
                NN.active=@(x) exp(-x.^2);
                NN.activeDerivate=@(x,a) -2*x.*a; 
            case 'Sigmoid'
                NN.active=@(x) 1./(1+exp(-x));
                NN.activeDerivate=@(a) a.*(1-a); 
            case 'tanh'
                NN.active=@(x) tanh(x);
                NN.activeDerivate=@(a) (1-a.^2);
            case 'ReLU'
                NN.active=@(x) max(0,x);
                NN.activeDerivate=@(x) Heaviside(x);
        end
    else
        NN.active=NN.ActivationFunction;
        
    end
    
    for i=1:NumOfLayer
        [W,b]=LayerInitialization(LayerMatrix(:,i));
        NN.weight{i}=W; NN.bias{i}=b;
        SizeW=size(W); SizeB=size(b);
        ZeroW=zeros(SizeW); ZeroB=zeros(SizeB);

        
        NN.FirstMomentW{i}=ZeroW;
        NN.FirstMomentB{i}=ZeroB;
        NN.SecondMomentW{i}=ZeroW;
        NN.SecondMomentB{i}=ZeroB;

    end

    if ResidualOn==1
        for i=2:NN.depth-1
            NN.ResMap{i}=IdentityMap(size(NN.bias{i},1),size(NN.bias{i-1},1));
        end
    end

end

function [W,b]=LayerInitialization(v)
    rng(1)
    InDim=v(1); OutDim=v(2);
    Radius=sqrt(6/(InDim+OutDim));
    temp=rand(OutDim,InDim);
    W=Radius*(temp-0.5*rand(OutDim,InDim));
    b=zeros(OutDim,1);

end

function out=SoftMax(x)
    Max=max(x);
    x=x-Max; % Improve numerical stability.
    u=exp(x);
    out=u./sum(u);
end

function ScalarClass=HotToIndex(OneHotVector)
    ScalarClass=zeros(1,size(OneHotVector,1));
    for i=1:size(OneHotVector,2)
        Class=find(OneHotVector(:,i));
        ScalarClass(i)=Class;
    end

end

function OneHotMatrix=IndexToHot(ScalarClass)
    NumOfClass=max(ScalarClass);
    NumOfData=size(ScalarClass,2);
    OneHotMatrix=zeros(NumOfClass,NumOfData);
    for i=1:NumOfData
        HotIndex=ScalarClass(i);
        OneHotMatrix(HotIndex,i)=1;
    end

end

function d=Heaviside(x)
    d=double(x>0);
end

function M=IdentityMap(Row,Column)
A=eye(Row,Column);

if Column>Row
    counter=0;
    for i=Row+1:Column
        counter=counter+1;
        if counter>Row
            counter=1;
        end
        A(counter,i)=1;
    end
end
M=sparse(A);
end