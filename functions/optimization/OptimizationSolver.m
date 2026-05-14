function optimizedNN = OptimizationSolver(data, label, NN, option)
%OPTIMIZATIONSOLVER  Train a neural network with stochastic or quasi-Newton solvers.

    if nargin < 4 || isempty(option)
        option = struct();
    end

    option = normalizeSolverOptions(option, NN, data);
    NN = prepareTrainingState(data, label, NN, option);
    label = applyLabelScaling(label, NN);

    optimizedNN = runSelectedSolver(data, label, NN, option);
    optimizedNN = attachPredictionApi(data, label, optimizedNN);
    printFinalReport(data, label, optimizedNN);
    optimizedNN = pruneOptimizerState(optimizedNN, option);
end

function option = normalizeSolverOptions(option, NN, data)
    if ~isfield(option, 'Solver') || isempty(option.Solver)
        if strcmp(NN.Cost, 'Entropy')
            option.Solver = 'ADAM';
        else
            option.Solver = 'Auto';
        end
    end
    option.Solver = char(option.Solver);

    if ~isfield(option, 's0') || isempty(option.s0)
        option.s0 = 2e-3;
    end
    if ~isfield(option, 'BatchSize') || isempty(option.BatchSize)
        option.BatchSize = max(1, round(size(data, 2) / 10));
    end
    if ~isfield(option, 'storeHistory')
        option.storeHistory = false;
    end
    if ~isfield(option, 'storeBfgs')
        option.storeBfgs = false;
    end
end

function NN = prepareTrainingState(data, label, NN, option)
    NN.numOfData = size(data, 2);
    NN.MeanFactor = getMeanFactor(NN.Cost, NN.numOfData);
    NN.Solver = option.Solver;

    NN = configureInputScaling(data, NN);
    NN = configureLabelScaling(label, NN);
    NN = configureWeights(option, NN);

    if ~isfield(NN, 'activeDerivate')
        warning('OptimizationSolver:MissingActivationDerivative', ...
            'Activation derivative is missing. Custom activations require NN.activeDerivate.');
    end
end

function meanFactor = getMeanFactor(costName, numData)
    switch costName
        case 'Entropy'
            meanFactor = 1 / numData;
        case 'MSE'
            meanFactor = 2 / numData;
        case 'MAE'
            meanFactor = 1 / numData;
        case 'SSE'
            meanFactor = 2;
        otherwise
            error('OptimizationSolver:UnknownCost', ...
                'Unknown cost "%s".', costName);
    end
end

function NN = configureInputScaling(data, NN)
    if strcmp(NN.InputAutoScaling, 'on')
        scale = std(data, 0, 2);
        scale(scale == 0) = 1;
        center = mean(data, 2);
        NN.InputCenterVector = center ./ scale;
        NN.InputScaleVector = 1 ./ scale;
    else
        NN.InputCenterVector = 0;
        NN.InputScaleVector = 1;
    end
end

function NN = configureLabelScaling(label, NN)
    if strcmp(NN.LabelAutoScaling, 'on')
        scale = std(label, 0, 2);
        scale(scale == 0) = 1;
        center = mean(label, 2);
        NN.LabelCenterVector = center;
        NN.LabelScaleVector = scale;
    else
        NN.LabelCenterVector = 0;
        NN.LabelScaleVector = 1;
    end
end

function scaledLabel = applyLabelScaling(label, NN)
    if strcmp(NN.LabelAutoScaling, 'on')
        scaledLabel = (label - NN.LabelCenterVector) ./ NN.LabelScaleVector;
    else
        scaledLabel = label;
    end
end

function NN = configureWeights(option, NN)
    if isfield(option, 'weighted')
        NN.SampleWeight = [];
        NN.Weighted = option.weighted;
        NN.WeightedFlag = 1;
    else
        NN.WeightedFlag = 0;
    end
end

function optimizedNN = runSelectedSolver(data, label, NN, option)
    switch option.Solver
        case {'BFGS', 'LBFGS'}
            optimizedNN = QuasiNewtonSolver(data, label, NN, option);
        case {'AdamW', 'ADAM', 'SGDM', 'SGD', 'RMSprop'}
            optimizedNN = StochasticSolver(data, label, NN, option);
        case 'Auto'
            optimizedNN = runAutoSolver(data, label, NN, option);
        otherwise
            error('OptimizationSolver:UnknownSolver', ...
                'Unknown solver "%s".', option.Solver);
    end
end

function optimizedNN = runAutoSolver(data, label, NN, option)
    totalIteration = 800;
    if isfield(option, 'MaxIteration')
        totalIteration = option.MaxIteration;
    end

    timer = tic;
    adamOption = option;
    adamOption.Solver = 'ADAM';
    adamOption.s0 = 2e-3;
    adamOption.MaxIteration = round(totalIteration / 4);
    adamOption.BatchSize = max(1, round(size(data, 2) / 10));
    NN = StochasticSolver(data, label, NN, adamOption);

    disp('------------------------------------------------------')
    disp(['First Stage Optimization Finished in  ', num2str(adamOption.MaxIteration), '  Iteration.'])
    disp('------------------------------------------------------')

    bfgsOption = option;
    bfgsOption.Solver = 'BFGS';
    bfgsOption.MaxIteration = totalIteration - adamOption.MaxIteration;
    optimizedNN = QuasiNewtonSolver(data, label, NN, bfgsOption);
    optimizedNN.OptimizationTime = toc(timer);
end

function NN = attachPredictionApi(data, label, NN)
    net = getNetworkHandle(NN);

    if strcmp(NN.LabelAutoScaling, 'on')
        NN.Evaluate = @(x) NN.LabelScaleVector .* net(x, NN) + NN.LabelCenterVector;
        residual = (NN.LabelScaleVector .* label + NN.LabelCenterVector) - NN.Evaluate(data);
    else
        NN.Evaluate = @(x) net(x, NN);
        residual = label - NN.Evaluate(data);
    end

    if strcmp(NN.Cost, 'Entropy')
        NN.ComputeAccuracy = @(d,l) ComputeAccuracy(d, l, NN);
        NN.Predict = @(d) ClassPredict(d, NN);
        NN.Accuracy = ComputeAccuracy(data, label, NN);
    else
        NN.Derivate = @(x) AutomaticDerivate(x, NN);
        NN.MeanAbsoluteError = sum(abs(residual), [1 2]) / NN.numOfData;
    end
end

function net = getNetworkHandle(NN)
    switch NN.NetworkType
        case 'ANN'
            net = @(x, nn) ANN(x, nn);
        case 'ResNet'
            net = @(x, nn) ResNet(x, nn);
        otherwise
            error('OptimizationSolver:UnsupportedNetworkType', ...
                'NetworkType must be ANN or ResNet.');
    end
end

function printFinalReport(data, label, NN)
    disp('------------------------------------------------------')
    fprintf('Max Iteration : %d , Cost : %16.8f \n', NN.Iteration, CostFunction(data, label, NN));

    if strcmp(NN.Cost, 'Entropy')
        fprintf('Accuracy : %6.2f %% \n', NN.Accuracy);
    else
        fprintf('Mean Absolute Error : %8.4f\n', NN.MeanAbsoluteError);
    end

    fprintf('Optimization Time : %5.1f\n', NN.OptimizationTime);
    disp('------------------------------------------------------')
end

function NN = pruneOptimizerState(NN, option)
    NN = dropFields(NN, {'SearchDirection', 'Gradient'});
    if ~option.storeBfgs
        NN = dropFields(NN, {'BFGS'});
    end

    if ~option.storeHistory
        NN = dropFields(NN, { ...
            'OptimizationHistory', 'StepSizeHistory', 'LineSearchIteration', ...
            'BatchCost', 'rho', 'CurvatureConditon', 'lbfgsState' ...
        });
    end
end

function s = dropFields(s, names)
    for idx = 1:numel(names)
        if isfield(s, names{idx})
            s = rmfield(s, names{idx});
        end
    end
end
