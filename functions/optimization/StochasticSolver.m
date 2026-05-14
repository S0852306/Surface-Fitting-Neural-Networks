function OptimizedNN = StochasticSolver(data,label,NN,option)
%STOCHASTICSOLVER  Stochastic optimization (SGD, ADAM, etc.)
%   This helper was extracted from OptimizationSolver to simplify the main
%   flow. It handles mini-batch looping, gradient selection and update rules.

Counter = 0;

% ---------------- select gradient solver ----------------
NetworkType = NN.NetworkType;
if ~isfield(option,'GradientSolver')
    switch NetworkType
        case 'ANN'
            AutoGrad = @(d,l,nn) AutomaticGradient(d,l,nn);
        case 'ResNet'
            AutoGrad = @(d,l,nn) ResNetAutomaticGradient(d,l,nn);
        case 'SirenNet'
            AutoGrad=@(d,l,nn) SirenNetAutomaticGradient(d,l,nn); 
    end
else
    GradientSolver = option.GradientSolver;
    switch NetworkType
        case 'ANN'
            switch GradientSolver
                case 'Element'
                    AutoGrad=@(d,l,nn) ElementWiseAG(d,l,nn);
                case 'Column'
                    AutoGrad=@(d,l,nn) ColumnWiseAG(d,l,nn);
                case 'General'
                    AutoGrad=@(d,l,nn) ComplexStepGradient(d,l,nn);
            end
        case 'ResNet'
            switch GradientSolver
                case 'Element'
                    AutoGrad=@(d,l,nn) ElementWiseRNAG(d,l,nn);
                case 'Column'
                    AutoGrad=@(d,l,nn) ColumnWiseRNAG(d,l,nn);
                case 'General'
                    AutoGrad=@(d,l,nn) ComplexStepGradient(d,l,nn);
            end
        case 'SirenNet'
            AutoGrad=@(d,l,nn) ComplexStepGradient(d,l,nn);

    end
end
% -------------------------------------------------------

BatchCost = zeros(2,1);
BatchSize = option.BatchSize;
tic
for j = 1:option.MaxIteration
    NN.Iteration = j;

    if option.BatchSize == NN.numOfData
        Sample.Data{1}  = data;
        Sample.Label{1} = label;
    else
        Sample = Shuffle(data,label,BatchSize);
    end

    for k = 1:numel(Sample.Label)
        Counter = Counter + 1;
        NN.StochasticCounter = Counter;
        ShuffledData = Sample.Data{k};
        ShuffledLabel = Sample.Label{k};
        if NN.WeightedFlag==1
            NN.SampleWeight = NN.Weighted(Sample.Index{k});
        end
        [dw,db,dc] = AutoGrad(ShuffledData,ShuffledLabel,NN);
        NN = StochasticUpdateRule(dw,db,NN,option,dc);
        BatchCost(Counter) = CostFunction(ShuffledData,ShuffledLabel,NN);
    end

    CurrentCost = CostFunction(data,label,NN);
    if rem(j,floor(option.MaxIteration/20)) == 0
        FormatSpec = 'Iteration : %d , Cost : %16.8f \n';
        fprintf(FormatSpec,j,CurrentCost);
    end
    NN.OptimizationHistory(j) = CurrentCost;
end

NN.OptimizationTime = toc;
NN.BatchCost = BatchCost;
OptimizedNN = NN;
end