function FunctionOutput=CostFunction(data,label,NN)
Cost=NN.Cost;
NetworkType=NN.NetworkType;
switch NetworkType
    case'ANN'
        Net=@(x,NN) ANN(x,NN);
    case 'ResNet'
        Net=@(x,NN) ResNet(x,NN);
end

predict=Net(data,NN);

switch Cost
    case 'SSE'
        temp=(label-predict).^2;
        E=sum(temp,[1 2]);
    case 'MSE'
        temp=(label-predict).^2;
        E=NN.MeanFactor*sum(temp,[1 2]);
    case 'Entropy'
        temp=-label.*log(max(predict,1e-8));
        E=NN.MeanFactor*sum(temp,[1 2]);
end
FunctionOutput=E;