function [dw,db,dc]=AutomaticGradient(data,label,NN)

data=NN.InputScaleVector.*data-NN.InputCenterVector;

splineOn = isfield(NN,'splineOn') && NN.splineOn;

v=data;

actID = NN.activationID;  % 1..5 built-in, 7 spline, 0 custom

for j=1:NN.depth-1
    z=NN.weight{j}*v+NN.bias{j};

    % ---- forward activation (fast) ----
    if splineOn
        if NN.bsplineOn
            [a, d] = BSplineActivation(z, NN.splineCoeff{j}, NN.bsplineGrid);
        else
            [a, d] = SplineActivation(z, NN.splineCoeff{j}, NN.splineGrid);
        end
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
    Memory.A{j}=a;
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

dw{NN.depth}=g*(Memory.A{NN.depth-1}.' );
db{NN.depth}=sum(g,2);

for j=NN.depth-1:-1:2
    dLda = (NN.weight{j+1}.')*g;          % dL/da_j
    if splineOn
        if NN.bsplineOn
            dc{j} = BSplineCoeffGrad(dLda, Memory.Z{j}, NN.bsplineGrid);
        else
            dc{j} = SplineCoeffGrad(dLda, Memory.Z{j}, NN.splineGrid);
        end
    end
    g=Memory.D{j}.*dLda;                  % dL/dz_j
    A=(Memory.A{j-1}).';
    dw{j}=g*A;
    db{j}=sum(g,2);
end

% Compute Gradient For First Layer
dLda = (NN.weight{2}.')*g;
if splineOn
    if NN.bsplineOn
        dc{1} = BSplineCoeffGrad(dLda, Memory.Z{1}, NN.bsplineGrid);
    else
        dc{1} = SplineCoeffGrad(dLda, Memory.Z{1}, NN.splineGrid);
    end
end
g=Memory.D{1}.*dLda;
A=data.';
dw{1}=g*A;
db{1}=sum(g,2);

end