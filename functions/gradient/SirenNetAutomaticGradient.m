function [dw, db] = SirenNetAutomaticGradient(data, label, NN)
% SirenNetAutomaticGradient (Hybrid SIREN)
% Backprop for SirenNet with only first K hidden layers using sine.
%
% IO unchanged:
%   [dw, db] = SirenNetAutomaticGradient(data, label, NN)

    % ------------------------------------------------------------
    % 0) Input scaling
    % ------------------------------------------------------------
    data = NN.InputScaleVector .* data - NN.InputCenterVector;
    v = data;

    % ------------------------------------------------------------
    % 0.5) Siren controls (safe defaults)
    % ------------------------------------------------------------
    K = 1;
    useOmega = false;
    omega0 = 30;
    omegaH = 1;

    if isfield(NN,'siren')
        if isfield(NN.siren,'numSineLayers') && ~isempty(NN.siren.numSineLayers)
            K = NN.siren.numSineLayers;
        end
        if isfield(NN.siren,'useOmegaInForward') && ~isempty(NN.siren.useOmegaInForward)
            useOmega = logical(NN.siren.useOmegaInForward);
        end
        if isfield(NN.siren,'omega0') && ~isempty(NN.siren.omega0)
            omega0 = NN.siren.omega0;
        end
        if isfield(NN.siren,'omegaHidden') && ~isempty(NN.siren.omegaHidden)
            omegaH = NN.siren.omegaHidden;
        end
    end

    % ------------------------------------------------------------
    % 0.6) activation selection for NON-sine layers
    % ------------------------------------------------------------
    actID = NN.activationID;  % 1..6 built-in, 0 custom

    % ------------------------------------------------------------
    % 1) Forward pass + cache (hidden layers)
    % ------------------------------------------------------------
    numHidden = NN.depth - 1;
    if numHidden < 0
        error('NN.depth must be >= 1');
    end

    % clamp K
    if K < 0, K = 0; end
    if K > numHidden, K = numHidden; end

    if numHidden >= 1
        for j = 1:numHidden
            z = NN.weight{j} * v + NN.bias{j};

            if j <= K
                % ---- SINE layers ----
                if useOmega
                    if j == 1
                        zj = omega0 * z;
                        a  = sin(zj);
                        d  = omega0 * cos(zj);
                    else
                        zj = omegaH * z;
                        a  = sin(zj);
                        d  = omegaH * cos(zj);
                    end
                else
                    a = sin(z);
                    d = cos(z);
                end
            else
                % ---- NON-SINE layers ----
                if actID ~= 0
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
                            a = (1 - z.^2) .* exp(-0.5 * z.^2);
                            d = exp(-0.5*z.^2) .* (z.^3 - 3*z);
                        case 6 % Sine
                            a = sin(z);
                            d = cos(z);
                        otherwise
                            error('Bad activationID');
                    end
                else
                    a = NN.customActive(z);
                    d = NN.customDer(z,a);
                end
            end

            v = a;
            Memory.A{j} = a;
            Memory.D{j} = d;  % da/dz
        end
        lastHiddenAct = Memory.A{numHidden};  %#ok<NASGU>
    end

    % ------------------------------------------------------------
    % 2) Output layer forward
    % ------------------------------------------------------------
    % output pre-activation uses v which is:
    %   - data, if numHidden==0
    %   - last hidden activation, if numHidden>=1
    zOut = NN.weight{NN.depth} * v + NN.bias{NN.depth};
    yhat = NN.OutActive(zOut);
    Memory.A{NN.depth} = yhat;

    % ------------------------------------------------------------
    % 3) Loss gradient wrt output
    % ------------------------------------------------------------
    if strcmp(NN.Cost,'MAE') == 1
        ErrorVector = NN.MeanFactor * sign(yhat - label);
    else
        ErrorVector = NN.MeanFactor * (yhat - label);
    end

    % ------------------------------------------------------------
    % 3.5) Optional weighting (robust)
    % ------------------------------------------------------------
    weightedOn = isfield(NN,'WeightedFlag') && NN.WeightedFlag == 1;
    if weightedOn
        % be defensive about NN.numOfData availability
        if isfield(NN,'numOfData') && size(label,2) == NN.numOfData && isfield(NN,'Weighted')
            DataWeightMatrix = NN.Weighted;
        elseif isfield(NN,'SampleWeight')
            DataWeightMatrix = NN.SampleWeight;
        else
            % fallback: treat as unweighted if weights not provided
            weightedOn = false;
        end
    end

    if ~weightedOn
        g = ErrorVector;
    else
        g = DataWeightMatrix .* ErrorVector;
    end

    % ------------------------------------------------------------
    % 4) Allocate grads
    % ------------------------------------------------------------
    dw = NN.weight;
    db = NN.bias;

    % ------------------------------------------------------------
    % 5) Gradient for output layer
    % ------------------------------------------------------------
    if numHidden >= 1
        Aprev = Memory.A{NN.depth-1};   % last hidden activation
    else
        Aprev = data;                  % no hidden -> input is data
    end

    dw{NN.depth} = g * (Aprev.');
    db{NN.depth} = sum(g, 2);

    % ------------------------------------------------------------
    % 6) Backprop through hidden layers if any
    % ------------------------------------------------------------
    if numHidden == 0
        return; % done
    end

    % hidden layers j = depth-1 ... 2
    for j = NN.depth-1:-1:2
        g = Memory.D{j} .* ((NN.weight{j+1}.') * g);
        A = (Memory.A{j-1}).';
        dw{j} = g * A;
        db{j} = sum(g, 2);
    end

    % first layer
    if NN.depth >= 2
        g = Memory.D{1} .* ((NN.weight{2}.') * g);
        A = data.';
        dw{1} = g * A;
        db{1} = sum(g, 2);
    end
end