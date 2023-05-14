function [dw,db]=ComplexStepGradient(data,label,NN)

% Numerical Algorithm Parameters Setting
% -----------------------------------------------------------
dwRecord=NN.weight;
dbRecord=NN.bias;
i=complex(0,1);


% OriginalCost=CostFunction(data,label,NN);
% fv: matrix ,row: integration index, column: data index
% -----------------------------------------------------------
Step=1e-30; ReciprocalStep=1/Step;
for j=1:NN.depth
    NumOfLocalWeight=NN.LayerStruct(1,j)*NN.LayerStruct(2,j);
    NumOfLocalBias=NN.LayerStruct(2,j);
    
    for k=1:NumOfLocalWeight
        % Partial Derivative Computaion Loop
        z0=NN.weight{j}(k); TempNN=NN;
        z=z0+i*Step;
        TempNN.weight{j}(k)=z;

        PerturbCost=CostFunction(data,label,TempNN);
        dwRecord{j}(k)=imag(PerturbCost);
    end
    
    for k=1:NumOfLocalBias
        z0=NN.bias{j}(k); TempNN=NN;
        z=z0+i*Step;
        TempNN.bias{j}(k)=z;
        PerturbCost=CostFunction(data,label,TempNN);
        dbRecord{j}(k)=imag(PerturbCost);

    end
    dwRecord{j}=dwRecord{j}*ReciprocalStep;
    dbRecord{j}=dbRecord{j}*ReciprocalStep;
end
% Positive Better
dw=dwRecord;
db=dbRecord;
end

