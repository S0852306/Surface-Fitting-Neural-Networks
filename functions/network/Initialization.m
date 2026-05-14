function NN = Initialization(LayerStruct, NN)
%INITIALIZATION  Build NN struct, set defaults, activation, and initialize params.
% Adds built-in activation: 'Wavelet' (Mexican hat) + 'Sine' (for SIREN).
%
% Built-in activations (string):
%   'Gaussian' | 'Sigmoid' | 'tanh' | 'ReLU' | 'Wavelet' | 'Sine'
%
% SirenNet:
%   NN.NetworkType = 'SirenNet'
%   Optional:
%     NN.siren.omega0      (default 30)
%     NN.siren.omegaHidden (default 1)
%
% NOTE:
%   For compatibility with existing backprop signature NN.activeDerivate(z,a),
%   we implement SIREN frequency via weight initialization scaling (equivalent
%   to using sin(omega*z) in forward).

    if nargin == 1
        NN = struct();
    end

    % ------------------------------------------------------------
    % Defaults preset
    % ------------------------------------------------------------
    if isfield(NN,'Default') == 1
        NN = struct();
        NN.ActivationFunction = 'Gaussian';
        NN.Cost              = 'MSE';
        NN.NetworkType        = 'ANN';
        NN.InputAutoScaling   = 'on';
        NN.LabelAutoScaling   = 'off';
    end

    if ~isfield(NN,'ActivationFunction');  NN.ActivationFunction = 'Gaussian'; end
    if ~isfield(NN,'Cost');                NN.Cost = 'MSE'; end
    if ~isfield(NN,'NetworkType');         NN.NetworkType = 'ANN'; end
    if ~isfield(NN,'InputAutoScaling');    NN.InputAutoScaling = 'off'; end
    if ~isfield(NN,'LabelAutoScaling');    NN.LabelAutoScaling = 'off'; end
    if ~isfield(NN,'LineSearcher');        NN.LineSearcher = 'BackTrack'; end

    NN.MeanFactor  = 1;
    NN.PreTrained  = 0;

    % ------------------------------------------------------------
    % Layer bookkeeping
    % LayerMatrix = [inDims; outDims]
    % ------------------------------------------------------------
    LayerMatrix           = zeros(2, numel(LayerStruct));
    LayerMatrix(1,:)      = LayerStruct(:).';
    LayerMatrix(2,1:end-1)= LayerStruct(2:end);
    NumOfLayer            = size(LayerMatrix,2) - 1;

    NN.depth          = NumOfLayer;
    NN.LayerStruct    = LayerMatrix;
    NN.numOfWeight    = LayerMatrix(1,:) * LayerMatrix(2,:)';
    NN.numOfBias      = sum(LayerMatrix(2,:));
    ResidualOn = strcmp(NN.NetworkType,'ResNet');
    SirenOn    = strcmp(NN.NetworkType,'SirenNet');

    % ------------------------------------------------------------
    % Learnable spline activation defaults
    % ------------------------------------------------------------
    SplineOn  = false;
    BSplineOn = false;
    if ischar(NN.ActivationFunction) || isstring(NN.ActivationFunction)
        SplineOn  = strcmp(NN.ActivationFunction, 'Spline');
        BSplineOn = strcmp(NN.ActivationFunction, 'BSpline');
    end
    NN.splineOn  = SplineOn || BSplineOn;   % dc flows for both types
    NN.bsplineOn = BSplineOn;

    if SplineOn
        if ~isfield(NN,'spline'); NN.spline = struct(); end
        if ~isfield(NN.spline,'numGrid') || isempty(NN.spline.numGrid)
            NN.spline.numGrid = 16;
        end
        if ~isfield(NN.spline,'gridRange') || isempty(NN.spline.gridRange)
            NN.spline.gridRange = [-4 4];
        end
        if ~isfield(NN.spline,'initShape') || isempty(NN.spline.initShape)
            NN.spline.initShape = 'Gaussian';
        end
        numKnots = NN.spline.numGrid + 1;
        gMin = NN.spline.gridRange(1);
        gMax = NN.spline.gridRange(2);
        NN.splineGrid.knots        = linspace(gMin, gMax, numKnots);
        NN.splineGrid.h            = (gMax - gMin) / NN.spline.numGrid;
        NN.splineGrid.numKnots     = numKnots;
        NN.splineGrid.numIntervals = NN.spline.numGrid;
        numCoeffs = numKnots;
        NN.numOfSpline = (NumOfLayer - 1) * numCoeffs;
    elseif BSplineOn
        if ~isfield(NN,'bspline'); NN.bspline = struct(); end
        if ~isfield(NN.bspline,'order') || isempty(NN.bspline.order)
            NN.bspline.order = 4;          % cubic B-spline
        end
        if ~isfield(NN.bspline,'numGrid') || isempty(NN.bspline.numGrid)
            NN.bspline.numGrid = 16;
        end
        if ~isfield(NN.bspline,'gridRange') || isempty(NN.bspline.gridRange)
            NN.bspline.gridRange = [-4 4];
        end
        if ~isfield(NN.bspline,'initShape') || isempty(NN.bspline.initShape)
            NN.bspline.initShape = 'Gaussian';
        end
        kOrd = NN.bspline.order;
        G    = NN.bspline.numGrid;
        gMin = NN.bspline.gridRange(1);
        gMax = NN.bspline.gridRange(2);
        breakpts = linspace(gMin, gMax, G + 1);
        augKnots = [repmat(gMin, 1, kOrd-1), breakpts, repmat(gMax, 1, kOrd-1)];
        numBasis = G + kOrd - 1;
        NN.bsplineGrid.knots     = augKnots;
        NN.bsplineGrid.order     = kOrd;
        NN.bsplineGrid.numBasis  = numBasis;
        NN.bsplineGrid.gridRange = [gMin gMax];
        numCoeffs = numBasis;
        NN.numOfSpline = (NumOfLayer - 1) * numCoeffs;
    else
        NN.numOfSpline = 0;
    end

    NN.numOfParameters = NN.numOfWeight + NN.numOfBias + NN.numOfSpline;

    % ------------------------------------------------------------
    % Siren params defaults (only used when SirenOn)
    % ------------------------------------------------------------
    if SirenOn
        if ~isfield(NN,'siren'); NN.siren = struct(); end
        if ~isfield(NN.siren,'omega0') || isempty(NN.siren.omega0)
            NN.siren.omega0 = 30;     % common SIREN default
        end
        if ~isfield(NN.siren,'omegaHidden') || isempty(NN.siren.omegaHidden)
            NN.siren.omegaHidden = 1; % common default (paper often uses 1)
        end

        % Force activation to Sine unless user explicitly provided custom handle
        if ~isa(NN.ActivationFunction,'function_handle')
            NN.ActivationFunction = 'Gaussian';
        end
    end

    % ------------------------------------------------------------
    % Output activation for classification
    % ------------------------------------------------------------
    if strcmp(NN.Cost,'Entropy') == 1
        NN.OutActive   = @(x) SoftMax(x);
        NN.IndexToHot  = @(x) IndexToHot(x);
        NN.HotToIndex  = @(x) HotToIndex(x);
    else
        NN.OutActive   = @(x) x;
    end

    % ------------------------------------------------------------
    % Activation
    % Derivative signature everywhere: da_dz = deriv(z,a)
    % ------------------------------------------------------------
    activation = NN.ActivationFunction;

    % defaults
    NN.activationID = 1;
    NN.customActive = [];
    NN.customDer    = [];

    if ~isa(activation,'function_handle')
        NN.activationID = mapActivationNameToID(activation);

        % compatibility handles
        NN.active         = @(z) activationForwardFast(NN.activationID, z, []);
        NN.activeDerivate = @(z,a) activationDerivFast(NN.activationID, z, a, []);

    else
        % Custom activation
        NN.activationID = 0;
        NN.customActive = activation;
        NN.ActivationFunction = 'Custom';

        % Custom derivative must exist
        if isfield(NN,'activeDerivate') && isa(NN.activeDerivate,'function_handle')
            userDer = NN.activeDerivate;
        elseif isfield(NN,'CustomActiveDerivate') && isa(NN.CustomActiveDerivate,'function_handle')
            userDer = NN.CustomActiveDerivate;
            NN.activeDerivate = userDer;
        else
            error('Initialization:CustomActivationNeedsDerivative', ...
                ['Custom ActivationFunction requires derivative handle.\n' ...
                 'Set NN.activeDerivate = @(z,a) ... (recommended)\n' ...
                 'or NN.CustomActiveDerivate = @(z,a) ...']);
        end

        % Wrap derivative into unified (z,a) signature
        nIn = safeNargin(userDer);
        if nIn == 2
            NN.customDer = @(z,a) userDer(z,a);
        elseif nIn == 1
            NN.customDer = @(z,a) userDer(a);
        else
            error('Initialization:BadDerivativeSignature', ...
                'Custom derivative must accept 1 input (a) or 2 inputs (z,a).');
        end

        NN.active         = @(z) NN.customActive(z);
        NN.activeDerivate = @(z,a) NN.customDer(z,a);
    end

    % ------------------------------------------------------------
    % Initialize weights / moments
    % ------------------------------------------------------------
    for i = 1:NumOfLayer

        if SirenOn
            [W,b] = LayerInitializationSiren( ...
                LayerMatrix(:,i), i, NumOfLayer, ...
                NN.siren.omega0, NN.siren.omegaHidden ...
            );
        else
            [W,b] = LayerInitialization(LayerMatrix(:,i));
        end

        NN.weight{i} = W;
        NN.bias{i}   = b;

        ZeroW = zeros(size(W));
        ZeroB = zeros(size(b));

        NN.FirstMomentW{i}  = ZeroW;
        NN.FirstMomentB{i}  = ZeroB;
        NN.SecondMomentW{i} = ZeroW;
        NN.SecondMomentB{i} = ZeroB;
    end

    % ------------------------------------------------------------
    % Spline / B-spline coefficient initialization + moments
    % ------------------------------------------------------------
    if SplineOn
        numKnots = NN.splineGrid.numKnots;
        knotVals = NN.splineGrid.knots(:);   % column vector
        initC = initSplineCoeffs(knotVals, NN.spline.initShape);
        for i = 1:NumOfLayer - 1
            NN.splineCoeff{i}     = initC;
            NN.FirstMomentC{i}    = zeros(numKnots, 1);
            NN.SecondMomentC{i}   = zeros(numKnots, 1);
        end
    elseif BSplineOn
        nB   = NN.bsplineGrid.numBasis;
        kOrd = NN.bsplineGrid.order;
        aug  = NN.bsplineGrid.knots;
        % Greville abscissae (natural B-spline coefficient sites)
        greville = zeros(nB, 1);
        for ii = 1:nB
            greville(ii) = mean(aug(ii+1 : ii+kOrd-1));
        end
        initC = initSplineCoeffs(greville, NN.bspline.initShape);
        for i = 1:NumOfLayer - 1
            NN.splineCoeff{i}     = initC;
            NN.FirstMomentC{i}    = zeros(nB, 1);
            NN.SecondMomentC{i}   = zeros(nB, 1);
        end
    end

    % ------------------------------------------------------------
    % Residual maps (fixed, not trainable)
    % ------------------------------------------------------------
    if ResidualOn == 1
        for i = 2:NN.depth-1
            NN.ResMap{i} = IdentityMap(size(NN.bias{i},1), size(NN.bias{i-1},1));
        end
    end
end

% ======================================================================
% Activation mapping
% ======================================================================
function actID = mapActivationNameToID(name)
    switch name
        case 'Gaussian'
            actID = 1;
        case 'Sigmoid'
            actID = 2;
        case 'tanh'
            actID = 3;
        case 'ReLU'
            actID = 4;
        case 'Wavelet'
            actID = 5;
        case 'Sine'     % NEW: for SIREN hidden layers
            actID = 6;
        case 'Spline'   % learnable linear spline
            actID = 7;
        case 'BSpline'  % learnable B-spline
            actID = 8;
        otherwise
            error('Initialization:UnknownActivation', ...
                'Unknown ActivationFunction = "%s".', name);
    end
end

% ======================================================================
% Local helper: activation forward/derivative (compatibility handles)
% ======================================================================
function a = activationForwardFast(actID, z, customAct)
    switch actID
        case 1 % Gaussian
            a = exp(-z.^2);

        case 2 % Sigmoid
            a = 1 ./ (1 + exp(-z));

        case 3 % tanh
            a = tanh(z);

        case 4 % ReLU
            a = max(0, z);

        case 5 % Wavelet (Mexican hat / Ricker)
            a = (1 - z.^2) .* exp(-0.5 * z.^2);

        case 6 % Sine
            a = sin(z);

        case 7 % Spline (placeholder – real forward uses SplineActivation)
            a = z;  % identity fallback; ANN/ResNet override this path

        case 8 % BSpline (placeholder – real forward uses BSplineActivation)
            a = z;

        case 0 % Custom
            a = customAct(z);

        otherwise
            error('Bad activationID');
    end
end

function d = activationDerivFast(actID, z, a, customDer)
    switch actID
        case 1 % Gaussian
            d = -2 * z .* a;

        case 2 % Sigmoid
            d = a .* (1 - a);

        case 3 % tanh
            d = 1 - a.^2;

        case 4 % ReLU
            d = double(z > 0);

        case 5 % Wavelet
            d = exp(-0.5 * z.^2) .* (z.^3 - 3*z);

        case 6 % Sine: d/dz sin(z) = cos(z)
            d = cos(z);

        case 7 % Spline (placeholder – real deriv computed in gradient file)
            d = ones(size(z));

        case 8 % BSpline (placeholder – real deriv computed in gradient file)
            d = ones(size(z));

        case 0 % Custom
            d = customDer(z, a);

        otherwise
            error('Bad activationID');
    end
end

function n = safeNargin(fh)
    try
        n = nargin(fh);
    catch
        n = 2;
    end
end

% ======================================================================
% Weight initialization
% ======================================================================
function [W,b] = LayerInitialization(v)
    rng(1);
    InDim  = v(1);
    OutDim = v(2);
    Radius = sqrt(6/(InDim+OutDim));

    temp = rand(OutDim, InDim);
    W    = Radius * (temp - 0.5*rand(OutDim, InDim));
    b    = zeros(OutDim, 1);
end

function [W,b] = LayerInitializationSiren(v, layerIdx, depth, omega0, omegaHidden)
% LayerInitializationSiren
% Implements SIREN-style init while keeping activation = sin(z) and derivative = cos(z)
% by absorbing omega into weight scaling for hidden layers.
%
% For layer < depth (hidden):
%   first layer: Wtilde ~ U(-1/In, +1/In), then W = omega0 * Wtilde
%   other hidden: Wtilde ~ U(-sqrt(6/In)/omegaHidden, +...), then W = omegaHidden * Wtilde
%                => effective W ~ U(-sqrt(6/In), +sqrt(6/In))
%
% For output layer (linear):
%   W ~ U(-sqrt(6/In)/omegaHidden, +sqrt(6/In)/omegaHidden)   (common SIREN choice)

    rng(1);

    InDim  = v(1);
    OutDim = v(2);

    isOutput = (layerIdx == depth);

    if ~isOutput
        if layerIdx == 1
            bound = 1 / InDim;
            Wtilde = (2*rand(OutDim, InDim) - 1) * bound;
            W = omega0 * Wtilde;                 % absorb omega0
        else
            bound = sqrt(6 / InDim) / omegaHidden;
            Wtilde = (2*rand(OutDim, InDim) - 1) * bound;
            W = omegaHidden * Wtilde;            % absorb omegaHidden
        end
    else
        % output linear layer (no sine), keep smaller bound
        bound = sqrt(6 / InDim) / omegaHidden;
        W = (2*rand(OutDim, InDim) - 1) * bound;
    end

    b = zeros(OutDim, 1);
end

% ======================================================================
% Original helpers
% ======================================================================
function out = SoftMax(x)
    Max = max(x);
    x = x - Max;
    u = exp(x);
    out = u ./ sum(u);
end

function ScalarClass = HotToIndex(OneHotVector)
    ScalarClass = zeros(1, size(OneHotVector,2));
    for i = 1:size(OneHotVector,2)
        Class = find(OneHotVector(:,i));
        ScalarClass(i) = Class;
    end
end

function OneHotMatrix = IndexToHot(ScalarClass)
    NumOfClass = max(ScalarClass);
    NumOfData  = size(ScalarClass,2);

    OneHotMatrix = zeros(NumOfClass, NumOfData);
    for i = 1:NumOfData
        HotIndex = ScalarClass(i);
        OneHotMatrix(HotIndex,i) = 1;
    end
end

function M = IdentityMap(Row, Column)
    A = eye(Row, Column);

    if Column > Row
        counter = 0;
        for i = Row+1:Column
            counter = counter + 1;
            if counter > Row
                counter = 1;
            end
            A(counter, i) = 1;
        end
    end

    M = sparse(A);
end

% ======================================================================
% Spline coefficient initializer
% ======================================================================
function c = initSplineCoeffs(knotVals, shape)
% Initialize spline coefficients to approximate a known activation shape.
    switch shape
        case 'Gaussian'
            c = exp(-knotVals.^2);
        case 'Sigmoid'
            c = 1 ./ (1 + exp(-knotVals));
        case 'tanh'
            c = tanh(knotVals);
        case 'ReLU'
            c = max(0, knotVals);
        case 'Identity'
            c = knotVals;
        case 'SiLU'
            c = knotVals ./ (1 + exp(-knotVals));
        case 'Wavelet'
            c = (1 - knotVals.^2) .* exp(-0.5 * knotVals.^2);
        otherwise
            c = exp(-knotVals.^2);   % fallback to Gaussian
    end
end