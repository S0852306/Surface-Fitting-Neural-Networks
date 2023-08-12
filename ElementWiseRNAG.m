function [dw,db]=ElementWiseRNAG(data,label,NN)

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
% -----------------------------------------------------------

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
        P=0;
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

    for k=1:NN.LayerStruct(2,j)
            % Row
            for m=1:NN.LayerStruct(1,j)
                    % Column

                    %Ap, Zp are perturbed Matrixs
                    Sp=S; Zp=Z;
                    PerturbVector=(i*Step)*A0(m,:);
                    Zp(k,:)=Z(k,:)+PerturbVector;

                    Sp(k,:)=LayerActive(Zp(k,:));
                    Ap=Sp+P;
                    PerturbCost=LocalCostFunction(Ap,Zp,label,j,NN);
                    dwRecord{j}(k,m)=imag(PerturbCost);
            end
                
    end

   for k=1:NN.LayerStruct(2,j)
        % Row
        Sp=S; Zp=Z;
        PerturbVector=(i*Step);
        Zp(k,:)=Z(k,:)+PerturbVector;
        Sp(k,:)=LayerActive(Zp(k,:));
        Ap=Sp+P;
        PerturbCost=LocalCostFunction(Ap,Zp,label,j,NN);
        dbRecord{j}(k)=imag(PerturbCost);

    end
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
temp=(label-ResINN(A,Z,LayerIndex,NN)).^2;
    switch Cost
        case 'SSE'
            E=sum(temp,[1 2]);
        case 'MSE'
            E=mean(temp,[1 2]);
     end
FunctionOutput=E;
end