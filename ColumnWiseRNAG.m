function [dw,db]=ColumnWiseRNAG(data,label,NN)
% For Ordinary Neural Nets (Multi Layers Perceptron)
% Numerical Mehtod Parameters Setting
% -----------------------------------------------------------
if strcmp(NN.InputAutoScaling,'on')==1
    data=NN.InputScaleVector.*data-NN.InputCenterVector;
end
i=complex(0,1);
Step=1e-30; ReciprocalStep=1/Step;
dwRecord=NN.weight; dbRecord=NN.bias;
Memory=Nets(data,NN);
NN.ResMap{NN.depth}=0;
NumOfData=size(label,2);
OutputDimension=NN.LayerStruct(1,end);
%------------------------------------------------------------

for j=1:NN.depth
    % A = A{k}
    A=Memory.A{j};
    S=Memory.S{j};
    Z=Memory.Z{j};
    %A0, Z0 are Previous States i.e. A{k-1}


    if j==1
        A0=data;
        P=sparse(size(A,1),size(A,2));
    elseif j==NN.depth
        P=sparse(OutputDimension,1);
        A0=Memory.A{j-1};
    else
        A0=Memory.A{j-1};
        Z0=Memory.Z{j-1};
        P=NN.ResMap{j}*Z0;
    end

    if j~=NN.depth
        LayerActive=@(x) NN.active(x);
    else
        LayerActive=@(x) NN.OutActive(x);
    end

    
    OutputSize=NN.LayerStruct(2,j);
    SPE=(i*Step)*speye(OutputSize);
    Zw=repmat(Z,1,OutputSize);
    if j==NN.depth
        Pw=0;
    else
        Pw=repmat(P,1,OutputSize);
    end
    Sw=repmat(S,1,OutputSize);
    Sp=Sw;
    for k=1:NN.LayerStruct(1,j)
        PerturbMatrix=kron(SPE,A0(k,:));

        Zp=Zw+PerturbMatrix;
        Sp=LayerActive(Zp);
        Ap=Sp+Pw;
        PerturbCost=LocalCostFunction(Ap,Zp,label,j,NN);
        dwRecord{j}(:,k)=imag(PerturbCost);
    end
    
    PerturbMatrix=kron(SPE,ones(1,NumOfData));

    Zp=Zw+PerturbMatrix;
    Sp=LayerActive(Zp);
    Ap=Sp+Pw;
    PerturbCost=LocalCostFunction(Ap,Zp,label,j,NN);
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
    Memory.S=NN.bias;
    Memory.Z=NN.bias;

    for j=1:NN.depth-1 
        z=NN.weight{j}*v+NN.bias{j};
        if j>1
            Active=NN.active(z);
            v=Active+NN.ResMap{j}*vp;
        else
            Active=NN.active(z);
            v=Active;
        end
        vp=z;
        Memory.A{j}=v;
        Memory.S{j}=Active;
        Memory.Z{j}=z;
                
    end
    z=NN.weight{NN.depth}*v+NN.bias{NN.depth};
    Memory.A{NN.depth}=NN.OutActive(z);
    Memory.S{NN.depth}=NN.OutActive(z);
    Memory.Z{NN.depth}=z;
    Function=Memory;
end


function Predict=ResINN(A,Z,LayerIndex,NN)
%  The values of Layer k is known
%  A = sigma(Zk), Z = Zk 
    v=A;
    vp=Z;

    if LayerIndex<=NN.depth-2 
        for j=LayerIndex+1:NN.depth-1
            temp=NN.weight{j}*v+NN.bias{j}; % Z{k+1}
            v=NN.active(temp)+NN.ResMap{j}*vp; % A{k+1}
            vp=temp;
        end
        Predict=NN.OutActive(NN.weight{NN.depth}*v+NN.bias{NN.depth});
    elseif LayerIndex==NN.depth-1
        Predict=NN.OutActive(NN.weight{NN.depth}*v+NN.bias{NN.depth});
    elseif LayerIndex==NN.depth
        Predict=NN.OutActive(v);
    end

end


function FunctionOutput=LocalCostFunction(A,Z,label,LayerIndex,NN)
Cost=NN.Cost;
NumOfData=size(label,2);
NumOfVariable=size(A,1); %Number of Variables
OutputDimension=NN.LayerStruct(1,end);
label=repmat(label,1,NumOfVariable);

temp=(label-ResINN(A,Z,LayerIndex,NN)).^2;
Tensor=nan(NumOfData,OutputDimension,NumOfVariable);
for j=1:NumOfVariable
    Tensor(:,:,j)=(temp(:,(j-1)*NumOfData+1:j*NumOfData)).';
end
    switch Cost
        case 'SSE'
            E=sum(Tensor,[1 2]);
            E=reshape(E,NumOfVariable,1);
        case 'MSE'
            E=sum(Tensor,[1 2]);
            E=reshape(E,NumOfVariable,1);
    end

FunctionOutput=E;
end