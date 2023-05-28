function [dw,db]=ElementWiseAG(data,label,NN)

% Numerical Mehtod Parameters Setting
% -----------------------------------------------------------
if strcmp(NN.InputAutoScaling,'on')==1
    data=NN.InputScaleVector.*data-NN.InputCenterVector;
end
i=complex(0,1);
Step=1e-30; ReciprocalStep=1/Step;
dwRecord=NN.weight;
dbRecord=NN.bias;
Memory=Nets(data,NN);
%------------------------------------------------------------

for j=1:NN.depth

    M=Memory.On{j};
    Z=Memory.Off{j};
    if j~=1     
            M0=Memory.On{j-1};
    else
            M0=data;
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
                    Temp=M;
                    PerturbVector=(i*Step)*M0(m,:);

                    Temp(k,:)=LayerActive(Z(k,:)+PerturbVector);

                    PerturbCost=LocalCostFunction(Temp,label,j,NN);
                    dwRecord{j}(k,m)=imag(PerturbCost);
            end
                
    end

   for k=1:NN.LayerStruct(2,j)
        % Row

        Temp=M;
        PerturbVector=(i*Step);
        if j~=NN.depth
            Temp(k,:)=NN.active(Z(k,:)+PerturbVector);
        else
            Temp(k,:)=NN.OutActive(Z(k,:)+PerturbVector);
        end        

        PerturbCost=LocalCostFunction(Temp,label,j,NN);
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
    Memory.On=NN.bias;
    Memory.Off=NN.bias;

    for j=1:NN.depth-1 
        temp=NN.weight{j}*v+NN.bias{j};
        Memory.Off{j}=temp;
        v=NN.active(temp);
        Memory.On{j}=v;
    end
    temp=NN.weight{NN.depth}*v+NN.bias{NN.depth};
    Memory.Off{NN.depth}=temp;
    Memory.On{NN.depth}=NN.OutActive(temp);
    Function=Memory;
end

function Predict=AINN(A,LayerIndex,NN)
    v=A;
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
temp=(label-AINN(A,LayerIndex,NN)).^2;
    switch Cost
        case 'SSE'
            E=sum(temp,[1 2]);
        case 'MSE'
            E=mean(temp,[1 2]);
     end
FunctionOutput=E;
end