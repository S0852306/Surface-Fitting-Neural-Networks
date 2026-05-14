function FittedModel = NeuralFit(data, label, ioDiemension)
% NeuralFit
% IO unchanged: FittedModel = NeuralFit(data, label, ioDiemension)

    % ------------------------------------------------------------
    % Parse IO dimensions
    % ------------------------------------------------------------
    InputDimension  = ioDiemension(1);
    OutputDimension = ioDiemension(2);

    % ------------------------------------------------------------
    % Network architecture (kept identical)
    % ------------------------------------------------------------
    LayerStruct = [InputDimension, 10, 10, 10, 10, OutputDimension];

    % ------------------------------------------------------------
    % Dimension adjustment (kept logic)
    % ------------------------------------------------------------
    data  = ensureRowIsFeature(data,  InputDimension);
    label = ensureRowIsFeature(label, OutputDimension);

    NumOfData = size(label, 2);

    % ------------------------------------------------------------
    % NN init (kept fields & calls)
    % ------------------------------------------------------------
    NN.InputAutoScaling = 'on';
    NN.LabelAutoScaling = 'on';
    NN = Initialization(LayerStruct, NN);

    % ------------------------------------------------------------
    % Options (centralized)
    % ------------------------------------------------------------
    option = getDefaultFitOptions(NumOfData);

    % ------------------------------------------------------------
    % Stage 1: ADAM
    % ------------------------------------------------------------
    printHeader(600); % NOTE: original prints 600 (even though MaxIteration=50)
    NN = OptimizationSolver(data, label, NN, option);

    disp('------------------------------------------------------')
    disp(['First Stage Optimization Finished in  ', num2str(option.MaxIteration), '  Iterations.'])
    disp('------------------------------------------------------')

    % ------------------------------------------------------------
    % Stage 2: BFGS
    % ------------------------------------------------------------
    option.Solver       = 'BFGS';
    option.MaxIteration = 550;

    NN = OptimizationSolver(data, label, NN, option);

    % ------------------------------------------------------------
    % Report + output
    % ------------------------------------------------------------
    NN.Report   = FittingReport(data, label, NN);
    FittedModel = NN;
end

% ======================================================================
% Helpers (local functions)
% ======================================================================

function X = ensureRowIsFeature(X, expectedDim)
    % Original behavior: transpose if row dimension doesn't match expected
    if size(X, 1) ~= expectedDim
        X = X.';
    end
end

function option = getDefaultFitOptions(NumOfData)
    option.Solver = 'ADAM';

    if NumOfData >= 200
        option.BatchSize = floor(NumOfData / 15);
    else
        option.BatchSize = floor(NumOfData / 5);
    end

    option.s0           = 2e-3;
    option.MaxIteration = 50;
end

function printHeader(iterToPrint)
    disp('------------------------------------------------------')
    disp(['Optimization will terminate after ', num2str(iterToPrint), ' Iterations.'])
    disp('------------------------------------------------------')
end