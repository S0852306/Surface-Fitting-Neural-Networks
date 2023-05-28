function Derivate=DirectionalDerivate(Step,Direction,data,label,NN)

% Numerical Algorithm Parameters Setting
% -----------------------------------------------------------
if strcmp(NN.InputAutoScaling,'on')==1
    data=NN.InputScaleVector.*data-NN.InputCenterVector;
end
i=complex(0,1);
h=1e-30;
ReciprocalH=1/h;
% fv: matrix ,row: integration index, column: data index
% -----------------------------------------------------------


for j=1:NN.depth

    NN.weight{j}=NN.weight{j}+(Step+i*h)*Direction.Weight{j};
    NN.bias{j}=NN.bias{j}+(Step+i*h)*Direction.Bias{j};
end

Derivate=imag(CostFunction(data,label,NN))*ReciprocalH;

end

