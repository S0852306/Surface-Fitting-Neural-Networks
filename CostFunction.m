function FunctionOutput=CostFunction(data,label,NN)
Cost=NN.Cost;
NetworkType=NN.NetworkType;
switch NetworkType
    case'ANN'
        Net=@(x,NN) ANN(x,NN);
    case 'ResNet'
        Net=@(x,NN) ResNet(x,NN);
end
temp=(label-Net(data,NN)).^2;
switch Cost
    case 'SSE'
        E=sum(temp,[1 2]);
    case 'MSE'
        E=mean(temp,[1 2]);
end
FunctionOutput=E;