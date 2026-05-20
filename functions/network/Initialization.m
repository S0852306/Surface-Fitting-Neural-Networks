function NN = Initialization(LayerStruct, NN)
%INITIALIZATION  Build NN struct, set defaults, activation, and initialize params.
% Adds built-in activation: 'Wavelet' (Mexican hat) + 'Sine'.
%
% Built-in activations (string):
%   'Gaussian' | 'Sigmoid' | 'tanh' | 'ReLU' | 'Wavelet' | 'Sine'

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
        NN.LabelAutoScaling   = 'on';
    end

    if ~isfield(NN,'ActivationFunction');  NN.ActivationFunction = 'Gaussian'; end
    if ~isfield(NN,'Cost');                NN.Cost = 'MSE'; end
    if ~isfield(NN,'NetworkType');         NN.NetworkType = 'ANN'; end
    if ~isfield(NN,'InputAutoScaling');    NN.InputAutoScaling = 'off'; end
    if ~isfield(NN,'LabelAutoScaling');    NN.LabelAutoScaling = 'off'; end
    if ~isfield(NN,'LineSearcher');        NN.LineSearcher = 'BackTrack'; end

    if ~strcmp(NN.NetworkType,'ANN') && ~strcmp(NN.NetworkType,'ResNet')
        error('Initialization:UnsupportedNetworkType', ...
            'NetworkType must be ANN or ResNet.');
    end

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

    % ------------------------------------------------------------
    % Learnable spline activation defaults
    % ------------------------------------------------------------
    BSplineOn = false;
    if ischar(NN.ActivationFunction) || isstring(NN.ActivationFunction)
        BSplineOn = strcmp(NN.ActivationFunction, 'BSpline');
    end
    NN.splineOn  = BSplineOn;
    NN.bsplineOn = BSplineOn;

    if BSplineOn
        if ~isfield(NN,'bspline'); NN.bspline = struct(); end
        % Allow shorthand: NN.BSplineInitShape → NN.bspline.initShape
        if isfield(NN,'BSplineInitShape') && ~isfield(NN.bspline,'initShape')
            NN.bspline.initShape = NN.BSplineInitShape;
        end
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
        NN.bsplineGrid.numGrid   = G;
        NN.bsplineGrid.invH      = G / (gMax - gMin);
        NN.bsplineGrid = precomputeBSplinePoly(NN.bsplineGrid);
        numCoeffs = numBasis;
        NN.numOfSpline = (NumOfLayer - 1) * numCoeffs;
    else
        NN.numOfSpline = 0;
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
        [W,b] = LayerInitialization(LayerMatrix(:,i));

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
    % B-spline coefficient initialization + moments
    % ------------------------------------------------------------
    if BSplineOn
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

    % ------------------------------------------------------------
    % Layer Normalization (optional, for ANN and ResNet)
    % ------------------------------------------------------------
    if ~isfield(NN,'LayerNorm'); NN.LayerNorm = 'off'; end
    % ResNet 預設 on，ANN 由外部決定（NeuralFit 已處理）
    if strcmpi(NN.NetworkType, 'ResNet') && ~isfield(NN, 'LayerNorm')
        NN.LayerNorm = 'on';
    end
    NN.layerNormOn = strcmp(NN.LayerNorm, 'on');

    if NN.layerNormOn
        numLN = 0;
        for i = 1:NumOfLayer - 1
            H = LayerMatrix(2, i);
            NN.lnGamma{i} = ones(H, 1);
            NN.lnBeta{i}  = zeros(H, 1);
            NN.FirstMomentLnG{i}  = zeros(H, 1);
            NN.SecondMomentLnG{i} = zeros(H, 1);
            NN.FirstMomentLnB{i}  = zeros(H, 1);
            NN.SecondMomentLnB{i} = zeros(H, 1);
            numLN = numLN + 2*H;
        end
        NN.numOfLN = numLN;
    else
        NN.numOfLN = 0;
    end

    NN.numOfParameters = NN.numOfWeight + NN.numOfBias + NN.numOfSpline + NN.numOfLN;
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
        case 'Sine'
            actID = 6;
        case 'BSpline'  % learnable B-spline
            actID = 7;
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

        case 7 % BSpline (placeholder – real forward uses BSplineActivation)
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

        case 7 % BSpline (placeholder – real deriv computed in gradient file)
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

% ======================================================================
% Precompute B-spline polynomial coefficients (called once at init)
% ======================================================================
function grid = precomputeBSplinePoly(grid)
%PRECOMPUTEBSPLINEPOLY  Build per-span Horner coefficients for B and dB.
%   For each interior span s = 0..numGrid-1, there are k non-zero basis
%   functions.  Each is a polynomial of degree k-1 in the local coordinate
%   u = (z - breakpoint_s) / h,  u in [0,1).
%
%   We sample each span at k Chebyshev-like points, evaluate the basis via
%   Cox-de Boor (one time only), then solve for the polynomial coefficients
%   via a Vandermonde system.
%
%   Stored:
%     grid.P   [numGrid x k x k]      — Horner coefficients for B
%     grid.dP  [numGrid x k x (k-1)]  — Horner coefficients for dB/dz
%                                        (derivative in z, not u)

    k    = grid.order;
    numG = grid.numGrid;
    gMin = grid.gridRange(1);
    invH = grid.invH;
    h    = 1 / invH;

    % Chebyshev nodes on [0,1] for better conditioning than equispaced
    idx  = (1:k)';
    uSample = 0.5 * (1 - cos(pi * (2*idx - 1) / (2*k)));

    % Vandermonde matrix: columns are u^{k-1}, u^{k-2}, ..., u, 1
    V = zeros(k, k);
    for p = 1:k
        V(:, p) = uSample.^(k - p);
    end

    P  = zeros(numG, k, k);        % basis polynomial coefficients
    dP = zeros(numG, k, k - 1);    % derivative polynomial coefficients

    % Temporary grid without P/dP for Cox-de Boor evaluation
    tmpGrid = rmfield(grid, intersect(fieldnames(grid), {'P','dP'}));

    for s = 0:numG-1
        % k sample points in physical z-space for this span
        zs = gMin + (s + uSample) * h;

        % Evaluate basis via Cox-de Boor at these k points
        Bfull = bsplineBasisCoxDeBoor(zs, tmpGrid);    % [k x numBasis]

        % Extract the k non-zero bases for span s (columns s+1 .. s+k)
        Blocal = Bfull(:, s+1 : s+k);                   % [k x k]

        % Solve V * polyCoeffs = Blocal  =>  polyCoeffs for each basis
        Pspan = (V \ Blocal)';                           % [k x k]
        P(s+1, :, :) = Pspan;

        % Derivative coefficients: d/dz = (1/h) * d/du
        % If poly is a_{k-1} u^{k-1} + ... + a_1 u + a_0
        % then d/du = (k-1)*a_{k-1} u^{k-2} + ... + a_1
        for i = 1:k
            pc = Pspan(i, :);                            % [1 x k]
            dp = pc(1:end-1) .* ((k-1):-1:1) * invH;    % chain rule
            dP(s+1, i, :) = dp;
        end
    end

    grid.P  = P;
    grid.dP = dP;
end

% ======================================================================
% Original Cox-de Boor (used only by precomputeBSplinePoly at init)
% ======================================================================
function B = bsplineBasisCoxDeBoor(z, grid)
    knots    = grid.knots;
    k        = grid.order;
    gMin     = grid.gridRange(1);
    gMax     = grid.gridRange(2);

    zv = z(:);
    N  = numel(zv);
    numKnots = numel(knots);
    nSpans   = numKnots - 1;

    zInd = max(gMin, min(gMax, zv));

    Bc = zeros(N, nSpans);
    for i = 1:nSpans
        if knots(i+1) > knots(i)
            Bc(:,i) = double(zInd >= knots(i) & zInd < knots(i+1));
        end
    end
    lastNZ = find(diff(knots) > 0, 1, 'last');
    Bc(zInd == gMax, lastNZ) = 1;

    for p = 2:k
        nB   = nSpans - p + 1;
        Bnew = zeros(N, nB);
        for i = 1:nB
            d1 = knots(i+p-1) - knots(i);
            d2 = knots(i+p)   - knots(i+1);
            w1 = 0;  w2 = 0;
            if d1 > 0
                w1 = (zv - knots(i)) / d1 .* Bc(:,i);
            end
            if d2 > 0
                w2 = (knots(i+p) - zv) / d2 .* Bc(:,i+1);
            end
            Bnew(:,i) = w1 + w2;
        end
        Bc = Bnew;
    end
    B = Bc;
end
