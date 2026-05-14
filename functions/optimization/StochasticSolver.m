function optimizedNN = StochasticSolver(data, label, NN, option)
%STOCHASTICSOLVER  Mini-batch first-order optimization.

    option = normalizeStochasticOptions(option, NN);
    autoGrad = selectAnalyticalGradient(NN, option);

    state.counter = 0;
    history = initStochasticHistory(option, NN);
    progressInterval = max(1, floor(option.MaxIteration / 20));

    tic
    for iter = 1:option.MaxIteration
        NN.Iteration = iter;
        batches = makeMiniBatches(data, label, option.BatchSize, NN.numOfData);

        for batchIdx = 1:numel(batches.Label)
            state.counter = state.counter + 1;
            NN.StochasticCounter = state.counter;

            batchData = batches.Data{batchIdx};
            batchLabel = batches.Label{batchIdx};
            NN = setCurrentSampleWeight(NN, batches, batchIdx);

            [dw, db, dc] = autoGrad(batchData, batchLabel, NN);
            NN = StochasticUpdateRule(dw, db, NN, option, dc);

            if history.store
                history.batchCost(state.counter) = CostFunction(batchData, batchLabel, NN);
            end
        end

        shouldPrint = rem(iter, progressInterval) == 0;
        if history.store || shouldPrint
            currentCost = CostFunction(data, label, NN);
        end
        if shouldPrint
            fprintf('Iteration : %d , Cost : %16.8f \n', iter, currentCost);
        end
        if history.store
            history.cost(iter) = currentCost;
        end
    end

    NN.OptimizationTime = toc;
    if history.store
        NN.OptimizationHistory = history.cost;
        NN.BatchCost = history.batchCost(1:state.counter);
    end

    optimizedNN = NN;
end

function option = normalizeStochasticOptions(option, NN)
    if ~isfield(option, 'BatchSize') || isempty(option.BatchSize)
        option.BatchSize = max(1, round(NN.numOfData / 10));
    end
    option.BatchSize = max(1, min(option.BatchSize, NN.numOfData));

    if ~isfield(option, 'MaxIteration') || isempty(option.MaxIteration)
        option.MaxIteration = 500;
    end
    if ~isfield(option, 's0') || isempty(option.s0)
        option.s0 = 2e-3;
    end
    if ~isfield(option, 'storeHistory')
        option.storeHistory = false;
    end
    if strcmp(option.Solver, 'AdamW') && ~isfield(option, 'Regulator')
        option.Regulator = 0;
    end
end

function autoGrad = selectAnalyticalGradient(NN, option)
    if isfield(option, 'GradientSolver') && ~strcmpi(option.GradientSolver, 'Analytical')
        error('StochasticSolver:UnsupportedGradientSolver', ...
            ['Numerical full-gradient solvers were removed from training. ', ...
             'Use the default analytical gradient solver.']);
    end

    switch NN.NetworkType
        case 'ANN'
            autoGrad = @(d,l,nn) AutomaticGradient(d,l,nn);
        case 'ResNet'
            autoGrad = @(d,l,nn) ResNetAutomaticGradient(d,l,nn);
        otherwise
            error('StochasticSolver:UnsupportedNetworkType', ...
                'NetworkType must be ANN or ResNet.');
    end
end

function history = initStochasticHistory(option, NN)
    history.store = option.storeHistory;
    if history.store
        batchesPerIteration = max(1, ceil(NN.numOfData / option.BatchSize));
        history.cost = zeros(option.MaxIteration, 1);
        history.batchCost = zeros(option.MaxIteration * batchesPerIteration, 1);
    else
        history.cost = [];
        history.batchCost = [];
    end
end

function batches = makeMiniBatches(data, label, batchSize, numData)
    if batchSize == numData
        batches.Data{1} = data;
        batches.Label{1} = label;
        batches.Index{1} = 1:numData;
    else
        batches = Shuffle(data, label, batchSize);
    end
end

function NN = setCurrentSampleWeight(NN, batches, batchIdx)
    if isfield(NN, 'WeightedFlag') && NN.WeightedFlag == 1
        NN.SampleWeight = NN.Weighted(batches.Index{batchIdx});
    end
end
