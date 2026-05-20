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
    lnOn = isfield(NN,'layerNormOn') && NN.layerNormOn;
    for i = 1:(NN.depth - 1)
        z = NN.weight{i} * v + NN.bias{i};  % pre-activation
        if splineOn
            v = BSplineActivation(z, NN.splineCoeff{i}, NN.bsplineGrid);
        else
            v = NN.active(z);                % activation
        end

        % Layer Normalization (per-sample, across neurons)
        if lnOn
            mu = mean(v, 1);
            sig2 = var(v, 1, 1) + 1e-5;
            v = NN.lnGamma{i} .* ((v - mu) ./ sqrt(sig2)) + NN.lnBeta{i};
        end
    end

    % ------------------------------------------------------------
    % 2) Output layer
    % ------------------------------------------------------------
    zOut = NN.weight{NN.depth} * v + NN.bias{NN.depth};
    FunctionOutput = NN.OutActive(zOut);

end