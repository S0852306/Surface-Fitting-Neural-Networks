function [dw, db, dc] = ResNetAutomaticGradient(data, label, NN)
    
    data= NN.InputScaleVector.*data - NN.InputCenterVector;

    depth = NN.depth;
    splineOn = isfield(NN,'splineOn') && NN.splineOn;

    V = cell(depth-1, 1);
    Z = cell(depth-1, 1);
    D = cell(depth-1, 1);

    actID = NN.activationID;

    v = data;
    for j = 1:depth-1
        z = NN.weight{j} * v + NN.bias{j};

        % ---- forward activation (fast) ----
        if splineOn
            if NN.bsplineOn
                [a, d] = BSplineActivation(z, NN.splineCoeff{j}, NN.bsplineGrid);
            else
                [a, d] = SplineActivation(z, NN.splineCoeff{j}, NN.splineGrid);
            end
        elseif actID ~= 0
            switch actID
                case 1 % Gaussian
                    a = exp(-z.^2);
                    d = -2*z.*a;
                case 2 % Sigmoid
                    a = 1./(1+exp(-z));
                    d = a.*(1-a);
                case 3 % tanh
                    a = tanh(z);
                    d = 1 - a.^2;
                case 4 % ReLU
                    a = max(0,z);
                    d = double(z>0);
                case 5 % Wavelet
                    d = exp(-0.5*z.^2) .* (z.^3 - 3*z);                 
                otherwise
                    error('Bad activationID');
            end
        else
            a = NN.customActive(z);
            d = NN.customDer(z,a);
        end

        if j > 1
            v = a + NN.ResMap{j} * Z{j-1};
        else
            v = a;
        end

        Z{j} = z;
        V{j} = v;
        D{j} = d;
    end

    logits = NN.weight{depth} * v + NN.bias{depth};
    y = NN.OutActive(logits);

    if strcmp(NN.Cost, 'MAE') == 1
        ErrorVector = NN.MeanFactor * sign(y - label);
    else
        ErrorVector = NN.MeanFactor * (y - label);
    end

    if NN.WeightedFlag == 1
        if size(label, 2) == NN.numOfData
            DataWeightMatrix = NN.Weighted;
        else
            DataWeightMatrix = NN.SampleWeight;
        end
        gOut = DataWeightMatrix .* ErrorVector;
    else
        gOut = ErrorVector;
    end

    dw = NN.weight;
    db = NN.bias;
    if splineOn; dc = cell(depth-1, 1); else; dc = {}; end

    dw{depth} = gOut * (V{depth-1}.' );
    db{depth} = sum(gOut, 2);

    % backprop state
    gV = (NN.weight{depth}.') * gOut;  % dL/dv_{depth-1}
    gZskip = 0;                       % extra gradient into z_j from skip

    for j = depth-1:-1:1
        temp_gV = gV;                 % save dL/dv_j for residual injection

        % ---- spline coefficient gradient (dL/da_j = temp_gV) ----
        if splineOn
            if NN.bsplineOn
                dc{j} = BSplineCoeffGrad(temp_gV, Z{j}, NN.bsplineGrid);
            else
                dc{j} = SplineCoeffGrad(temp_gV, Z{j}, NN.splineGrid);
            end
        end

        gZ = gZskip + D{j} .* temp_gV;

        if j == 1
            vPrev = data;
        else
            vPrev = V{j-1};
        end

        dw{j} = gZ * (vPrev.');
        db{j} = sum(gZ, 2);

        if j > 1
            gV = (NN.weight{j}.') * gZ;
            gZskip = (NN.ResMap{j}.') * temp_gV;
        end
    end
end