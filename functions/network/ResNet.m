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

    for i = 1:(NN.depth - 1)
        z = NN.weight{i} * v + NN.bias{i};

        if splineOn
            if NN.bsplineOn
                a = BSplineActivation(z, NN.splineCoeff{i}, NN.bsplineGrid);
            else
                a = SplineActivation(z, NN.splineCoeff{i}, NN.splineGrid);
            end
        else
            a = NN.active(z);
        end

        if i > 1
            v = a + NN.ResMap{i} * vp;
        else
            v = a;
        end

        vp = z;
    end

    % ------------------------------------------------------------
    % 2) Output layer
    % ------------------------------------------------------------
    zOut = NN.weight{NN.depth} * v + NN.bias{NN.depth};
    FunctionOutput = NN.OutActive(zOut);
end