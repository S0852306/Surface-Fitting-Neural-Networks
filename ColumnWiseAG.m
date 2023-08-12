function [dw,db]=ColumnWiseAG(data,label,NN)
% For Ordinary Neural Nets (Multi Layers Perceptron)
% Numerical Method Parameters Setting
% -----------------------------------------------------------
if strcmp(NN.InputAutoScaling,'on')==1
    data=NN.InputScaleVector.*data-NN.InputCenterVector;
end
i=complex(0,1);
Step=1e-30; ReciprocalStep=1/Step;
Memory=Nets(data,NN);
NumOfData=size(label,2);
dwRecord=NN.weight;
dbRecord=NN.bias;
% -----------------------------------------------------------

for j=1:NN.depth


    Z=Memory.Z{j};
    if j~=1     
            A0=Memory.A{j-1};
    else
            A0=data;
    end
    
    if j~=NN.depth
        LayerActive=@(x) NN.active(x);
    else
        LayerActive=@(x) NN.OutActive(x);
    end
    
    OutputSize=NN.LayerStruct(2,j);
    SPE=(i*Step)*speye(OutputSize);
    Zw=repmat(Z,1,OutputSize);

    for k=1:NN.LayerStruct(1,j)
        PerturbMatrix=kron(SPE,A0(k,:));
        Zp=Zw+PerturbMatrix;

        Ap=LayerActive(Zp);
        PerturbCost=LocalCostFunction(Ap,label,j,NN);
        dwRecord{j}(:,k)=imag(PerturbCost);
    end

    PerturbMatrix=kron(SPE,ones(1,NumOfData));
    Zp=Zw+PerturbMatrix;
    Ap=LayerActive(Zp);
    PerturbCost=LocalCostFunction(Ap,label,j,NN);
    dbRecord{j}=imag(PerturbCost);

    dwRecord{j}=dwRecord{j}*ReciprocalStep;
    dbRecord{j}=dbRecord{j}*ReciprocalStep;
end

dw=dwRecord;
db=dbRecord;

end

function Function=Nets(data,NN)

    v=data;
    Memory.A=NN.bias;
    Memory.Z=NN.bias;

    for j=1:NN.depth-1 
        temp=NN.weight{j}*v+NN.bias{j};
        Memory.Z{j}=temp;
        v=NN.active(temp);
        Memory.A{j}=v;
    end
    temp=NN.weight{NN.depth}*v+NN.bias{NN.depth};
    Memory.Z{NN.depth}=temp;
    Memory.A{NN.depth}=NN.OutActive(temp);
    Function=Memory;
end

function Predict=AINN(data,LayerIndex,NN)
    v=data;


    if LayerIndex<=NN.depth-2
        for j=LayerIndex+1:NN.depth-1
            v=NN.active(NN.weight{j}*v+NN.bias{j});
        end
        Predict=NN.OutActive(NN.weight{NN.depth}*v+NN.bias{NN.depth});
    elseif LayerIndex==NN.depth-1
        Predict=NN.OutActive(NN.weight{NN.depth}*v+NN.bias{NN.depth});
    elseif LayerIndex==NN.depth
        Predict=v;
    end

end

function FunctionOutput=LocalCostFunction(A,label,LayerIndex,NN)
Cost=NN.Cost;

NumOfData=size(label,2);
NumOfVariable=size(A,1); %Number of Variables
OutputDimension=NN.LayerStruct(1,end);
label=repmat(label,1,NumOfVariable);

temp=(label-AINN(A,LayerIndex,NN)).^2;
Tensor=nan(NumOfData,OutputDimension,NumOfVariable);
for j=1:NumOfVariable
    Tensor(:,:,j)=(temp(:,(j-1)*NumOfData+1:j*NumOfData)).';
end

    switch Cost
        case 'SSE'
            E=sum(Tensor,[1 2]);
            E=reshape(E,NumOfVariable,1);
        case 'MSE'
            E=mean(Tensor,[1 2]);
            E=reshape(E,NumOfVariable,1);
     end
FunctionOutput=E;
end

