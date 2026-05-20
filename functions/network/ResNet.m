function FunctionOutput = ResNet(data, NN)
% ResNet forward pass
% IO unchanged: FunctionOutput = ResNet(data, NN)

    % ------------------------------------------------------------
    % 0) input autoscaling
    % ------------------------------------------------------------
    data = NN.InputScaleVector .* data - NN.InputCenterVector;


    % ------------------------------------------------------------
    % 1) Forward propagation with residual connections
    % ------------------------------------------------------------
    v = data;

    % vp stores the previous pre-activation z (used by ResMap for i>1)
    vp = [];
    splineOn = isfield(NN,'splineOn') && NN.splineOn;
    lnOn = isfield(NN,'layerNormOn') && NN.layerNormOn;

    for i = 1:(NN.depth - 1)
        z = NN.weight{i} * v + NN.bias{i};

        if splineOn
            a = BSplineActivation(z, NN.splineCoeff{i}, NN.bsplineGrid);
        else
            a = NN.active(z);
        end

        if i > 1
            v = a + NN.ResMap{i} * vp;
        else
            v = a;
        end

        % Layer Normalization (per-sample, across neurons)
        if lnOn
            mu = mean(v, 1);            % 1 x N
            sig2 = var(v, 1, 1) + 1e-5; % 1 x N
            v = NN.lnGamma{i} .* ((v - mu) ./ sqrt(sig2)) + NN.lnBeta{i};
        end

        vp = z;
    end

    % ------------------------------------------------------------
    % 2) Output layer
    % ------------------------------------------------------------
    zOut = NN.weight{NN.depth} * v + NN.bias{NN.depth};
    FunctionOutput = NN.OutActive(zOut);
end