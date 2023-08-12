function Derivate=AutomaticDerivate(x,NN)
% This is not the function used to train neural networks. 
% The gradients of network parameters are computed using "ElementWiseAG", "ElementWiseRANG", and so on.
i=complex(0,1); h=1e-30; rh=1/h;
NumOfVariable=size(x,1);
NumOfData=size(x,2);
NumOfOutput=NN.LayerStruct(1,end);
DerivateTensor=zeros(NumOfVariable,NumOfData,NumOfOutput);
for j=1:NumOfVariable
    xPerturb=x;
    xPerturb(j,:)=x(j,:)+i*h;
    DerivateMatrix=rh*imag(NN.Evaluate(xPerturb));
    for k=1:NumOfOutput
        DerivateTensor(j,:,k)=DerivateMatrix(k,:);
    end

end
% DerivateTensor = Modified Jacobian matrix for easier programing.
if NumOfData==1
    DerivateTensor=reshape(DerivateTensor,NumOfVariable,NumOfOutput);
    % Standard Jacobian matrix.
end

if NumOfOutput==1
    DerivateTensor=reshape(DerivateTensor,NumOfVariable,NumOfData);
    % Gradient, for all data.
end

Derivate=DerivateTensor;
%Derivate{i} i-th function output.
%Derivate{i}(k,:) \partial x_k for i-th function output, for all data point. 
end