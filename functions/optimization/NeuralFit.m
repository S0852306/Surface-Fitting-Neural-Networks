function fittedModel = NeuralFit(data, label, ioDimension, varargin)
% NeuralFit
%   fittedModel = NeuralFit(data, label, ioDimension)
%   fittedModel = NeuralFit(..., 'activation', 'Gaussian')
%
% Default activation is learnable quadratic B-spline:
%   NN.bspline.order = 3, NN.bspline.numGrid = 8.
%
% For smooth fitting, specify a fixed activation:
%   NeuralFit(data, label, dims, 'activation', 'Gaussian')
%   NeuralFit(data, label, dims, 'activation', 'Wavelet')

    config = parseFitConfig(varargin{:});

    inputDimension  = ioDimension(1);
    outputDimension = ioDimension(2);
    layerStruct = [inputDimension, 10, 10, 10, 10, outputDimension];

    data  = ensureRowIsFeature(data,  inputDimension);
    label = ensureRowIsFeature(label, outputDimension);
    numData = size(label, 2);

    NN.InputAutoScaling = 'on';
    NN.LabelAutoScaling = 'on';
    NN.NetworkType = 'ANN';
    NN.Cost = 'MSE';
    NN = applyBasisConfig(NN, config);
    NN = Initialization(layerStruct, NN);

    option = getDefaultFitOptions(numData, config);

    printHeader(option.stage1Iterations + option.stage2Iterations);
    option.Solver = 'ADAM';
    option.MaxIteration = option.stage1Iterations;
    NN = OptimizationSolver(data, label, NN, option);

    disp('------------------------------------------------------')
    disp(['First Stage Optimization Finished in  ', num2str(option.MaxIteration), '  Iterations.'])
    disp('------------------------------------------------------')

    option.Solver = 'BFGS';
    option.MaxIteration = option.stage2Iterations;
    NN = OptimizationSolver(data, label, NN, option);

    NN.Report = FittingReport(data, label, NN);
    fittedModel = NN;
end

function config = parseFitConfig(varargin)
    config.activation = 'BSpline';
    config.stage1Iterations = 50;
    config.stage2Iterations = 550;
    config.storeHistory = false;

    if mod(numel(varargin), 2) ~= 0
        error('NeuralFit:BadNameValue', 'Options must be name-value pairs.');
    end

    for idx = 1:2:numel(varargin)
        name = lower(string(varargin{idx}));
        value = varargin{idx + 1};
        switch name
            case {"basis", "activation"}
                config.activation = char(value);
            case "maxiteration"
                config.stage1Iterations = max(1, round(double(value) / 12));
                config.stage2Iterations = max(1, double(value) - config.stage1Iterations);
            case "stage1iterations"
                config.stage1Iterations = double(value);
            case "stage2iterations"
                config.stage2Iterations = double(value);
            case "storehistory"
                config.storeHistory = logical(value);
            otherwise
                error('NeuralFit:UnknownOption', 'Unknown option "%s".', varargin{idx});
        end
    end
end

function NN = applyBasisConfig(NN, config)
    actName = config.activation;

    if strcmpi(actName, 'BSpline')
        NN.ActivationFunction = 'BSpline';
        NN.bspline.order = 3;
        NN.bspline.numGrid = 8;
        NN.bspline.initShape = 'Gaussian';
    else
        NN.ActivationFunction = normalizeActivationName(actName);
    end
end

function actName = normalizeActivationName(actName)
    knownAct = {'Gaussian', 'Sigmoid', 'tanh', 'ReLU', 'Wavelet', 'Sine'};
    for idx = 1:numel(knownAct)
        if strcmpi(actName, knownAct{idx})
            actName = knownAct{idx};
            return
        end
    end
    error('NeuralFit:UnknownActivation', ...
        'Unknown activation "%s". Use BSpline, Gaussian, Sigmoid, tanh, ReLU, Wavelet, or Sine.', actName);
end

function x = ensureRowIsFeature(x, expectedDim)
    if size(x, 1) ~= expectedDim
        x = x.';
    end
end

function option = getDefaultFitOptions(numData, config)
    if numData >= 200
        option.BatchSize = max(1, floor(numData / 15));
    else
        option.BatchSize = max(1, floor(numData / 5));
    end

    option.s0 = 2e-3;
    option.storeHistory = config.storeHistory;
    option.stage1Iterations = config.stage1Iterations;
    option.stage2Iterations = config.stage2Iterations;
end

function printHeader(iterToPrint)
    disp('------------------------------------------------------')
    disp(['Optimization will terminate after ', num2str(iterToPrint), ' Iterations.'])
    disp('------------------------------------------------------')
end
