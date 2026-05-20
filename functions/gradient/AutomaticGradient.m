function [dw,db,dc,dlnG,dlnB]=AutomaticGradient(data,label,NN)

data=NN.InputScaleVector.*data-NN.InputCenterVector;

splineOn = isfield(NN,'splineOn') && NN.splineOn;
lnOn = isfield(NN,'layerNormOn') && NN.layerNormOn;

if lnOn
    LN_vhat = cell(NN.depth-1, 1);
    LN_invStd = cell(NN.depth-1, 1);
end

v=data;

actID = NN.activationID;  % 1..6 built-in, 7 BSpline, 0 custom

for j=1:NN.depth-1
    z=NN.weight{j}*v+NN.bias{j};

    % ---- forward activation (fast) ----
    if splineOn
        [a, d] = BSplineActivation(z, NN.splineCoeff{j}, NN.bsplineGrid);
        Memory.Z{j} = z;   % store pre-activation for coefficient gradient
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
                a = (1 - z.^2) .* exp(-0.5 * z.^2);
                d = exp(-0.5*z.^2) .* (z.^3 - 3*z);
            otherwise
                error('Bad activationID');
        end
    else
        a = NN.customActive(z);
        d = NN.customDer(z,a);
    end

    v=a;

    % Layer Normalization
    if lnOn
        H = size(v, 1);
        mu = mean(v, 1);
        sig2 = var(v, 1, 1) + 1e-5;
        invStd = 1 ./ sqrt(sig2);
        vhat = (v - mu) .* invStd;
        v = NN.lnGamma{j} .* vhat + NN.lnBeta{j};
        LN_vhat{j} = vhat;
        LN_invStd{j} = invStd;
    end

    Memory.A{j}=v;
    Memory.D{j}=d;
end

z=NN.weight{NN.depth}*v+NN.bias{NN.depth};

Memory.A{NN.depth}=NN.OutActive(z);
Memory.D{NN.depth}=z;

if strcmp(NN.Cost,'MAE')==1
    ErrorVector=NN.MeanFactor*sign(Memory.A{NN.depth}-label);
else
    ErrorVector=NN.MeanFactor*(Memory.A{NN.depth}-label);
end

if size(label,2)==NN.numOfData && NN.WeightedFlag==1
    DataWeightMatrix=NN.Weighted;
elseif size(label,2)~=NN.numOfData && NN.WeightedFlag==1
    DataWeightMatrix=NN.SampleWeight;
end

% Compute Gradient For Last Layer
if NN.WeightedFlag==0
    g=ErrorVector;
else
    g=DataWeightMatrix.*ErrorVector;
end
dw=NN.weight; db=NN.bias;
if splineOn; dc = cell(NN.depth-1, 1); else; dc = {}; end
if lnOn; dlnG = cell(NN.depth-1, 1); dlnB = cell(NN.depth-1, 1);
else; dlnG = {}; dlnB = {}; end

dw{NN.depth}=g*(Memory.A{NN.depth-1}.' );
db{NN.depth}=sum(g,2);

for j=NN.depth-1:-1:2
    gV = (NN.weight{j+1}.')*g;          % dL/dv_j (post-LN)

    % ---- LayerNorm backprop ----
    if lnOn
        H = size(gV, 1);
        vhat_j = LN_vhat{j};
        invStd_j = LN_invStd{j};
        gamma_j = NN.lnGamma{j};

        dlnG{j} = sum(gV .* vhat_j, 2);
        dlnB{j} = sum(gV, 2);

        dxhat = gV .* gamma_j;
        gV = (1/H) .* invStd_j .* (H * dxhat ...
            - sum(dxhat, 1) ...
            - vhat_j .* sum(dxhat .* vhat_j, 1));
    end

    if splineOn
        dc{j} = BSplineCoeffGrad(gV, Memory.Z{j}, NN.bsplineGrid);
    end
    g=Memory.D{j}.*gV;                  % dL/dz_j
    A=(Memory.A{j-1}).';
    dw{j}=g*A;
    db{j}=sum(g,2);
end

% Compute Gradient For First Layer
gV = (NN.weight{2}.')*g;

% ---- LayerNorm backprop for layer 1 ----
if lnOn
    H = size(gV, 1);
    vhat_j = LN_vhat{1};
    invStd_j = LN_invStd{1};
    gamma_j = NN.lnGamma{1};

    dlnG{1} = sum(gV .* vhat_j, 2);
    dlnB{1} = sum(gV, 2);

    dxhat = gV .* gamma_j;
    gV = (1/H) .* invStd_j .* (H * dxhat ...
        - sum(dxhat, 1) ...
        - vhat_j .* sum(dxhat .* vhat_j, 1));
end

if splineOn
    dc{1} = BSplineCoeffGrad(gV, Memory.Z{1}, NN.bsplineGrid);
end
g=Memory.D{1}.*gV;
A=data.';
dw{1}=g*A;
db{1}=sum(g,2);

end