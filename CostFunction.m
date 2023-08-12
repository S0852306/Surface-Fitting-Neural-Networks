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
if size(label,2)==NN.numOfData && NN.WeightedFlag==1
    DataWeightMatrix=NN.Weighted;
elseif size(label,2)~=NN.numOfData && NN.WeightedFlag==1
    DataWeightMatrix=NN.SampleWeight;
end

switch Cost
    case 'SSE'
        if isfield(NN,'Weighted')~=1
            temp=(label-predict).^2;
        else
            temp=DataWeightMatrix.*(label-predict).^2;
        end
        E=sum(temp,[1 2]);
    case 'MSE'
        if isfield(NN,'Weighted')~=1
            temp=(label-predict).^2;
        else
            temp=DataWeightMatrix.*(label-predict).^2;
        end
        E=NN.MeanFactor*sum(temp,[1 2]);
    case 'MAE'
        if isfield(NN,'Weighted')~=1
            temp=abs(label-predict);
        else
            temp=DataWeightMatrix.*abs(label-predict);
        end
        E=NN.MeanFactor*sum(temp,[1 2]);
        
    case 'Entropy'
        temp=-label.*log(max(predict,1e-8));
        E=NN.MeanFactor*sum(temp,[1 2]);
end
FunctionOutput=E;
