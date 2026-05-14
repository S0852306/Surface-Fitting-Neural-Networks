function FunctionOutput = ANN(data, NN)
% ANN forward pass
% IO unchanged: FunctionOutput = ANN(data, NN)

    % ------------------------------------------------------------
    % 0) Input scaling (always applied; identity if scaling off)
    % ------------------------------------------------------------
    v = NN.InputScaleVector .* data - NN.InputCenterVector;

    % ------------------------------------------------------------
    % 1) Hidden layers
    % ------------------------------------------------------------
    splineOn = isfield(NN,'splineOn') && NN.splineOn;
    for i = 1:(NN.depth - 1)
        z = NN.weight{i} * v + NN.bias{i};  % pre-activation
        if splineOn
            if NN.bsplineOn
                v = BSplineActivation(z, NN.splineCoeff{i}, NN.bsplineGrid);
            else
                v = SplineActivation(z, NN.splineCoeff{i}, NN.splineGrid);
            end
        else
            v = NN.active(z);                % activation
        end
    end

    % ------------------------------------------------------------
    % 2) Output layer
    % ------------------------------------------------------------
    zOut = NN.weight{NN.depth} * v + NN.bias{NN.depth};
    FunctionOutput = NN.OutActive(zOut);

end