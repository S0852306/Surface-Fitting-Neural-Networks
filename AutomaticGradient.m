function [dw,db]=AutomaticGradient(data,label,NN)
if strcmp(NN.InputAutoScaling,'on')==1
    data=NN.InputScaleVector.*data-NN.InputCenterVector;
end

v=data;

for j=1:NN.depth-1
    z=NN.weight{j}*v+NN.bias{j};
    v=NN.active(z);
    Memory.A{j}=v;
    if strcmp(NN.ActivationFunction,'Gaussian')
        Memory.D{j}=NN.activeDerivate(z,v);
    elseif strcmp(NN.ActivationFunction,'ReLU')
        Memory.D{j}=NN.activeDerivate(z);
    else
        Memory.D{j}=NN.activeDerivate(v);
    end
end

z=NN.weight{NN.depth}*v+NN.bias{NN.depth};

Memory.A{NN.depth}=NN.OutActive(z);
Memory.D{NN.depth}=z;
ErrorVector=2*NN.MeanFactor*(label-Memory.A{NN.depth});
% Compute Gradient For Last Layer
g=-ErrorVector;
dw=NN.weight; db=NN.bias;
dw{NN.depth}=g*(Memory.A{NN.depth-1}.');
for j=NN.depth-1:-1:2

    g=Memory.D{j}.*((NN.weight{j+1}.')*g);
    A=(Memory.A{j-1}).';
    dw{j}=g*A;
    db{j}=sum(g,2);
end
% Compute Gradient For First Layer
g=Memory.D{1}.*((NN.weight{2}.')*g);
A=data.';
dw{1}=g*A;
db{1}=sum(g,2);

end