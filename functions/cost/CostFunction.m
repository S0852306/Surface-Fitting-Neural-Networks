function costValue = CostFunction(data, label, NN)
%COSTFUNCTION  Evaluate the configured training objective.

    prediction = evaluateNetwork(data, NN);
    residual = label - prediction;

    switch NN.Cost
        case 'SSE'
            costValue = sum(applySampleWeight(residual.^2, label, NN), [1 2]);
        case 'MSE'
            costValue = NN.MeanFactor * sum(applySampleWeight(residual.^2, label, NN), [1 2]);
        case 'MAE'
            costValue = NN.MeanFactor * sum(applySampleWeight(abs(residual), label, NN), [1 2]);
        case 'Entropy'
            entropyTerm = -label .* log(max(prediction, 1e-8));
            costValue = NN.MeanFactor * sum(entropyTerm, [1 2]);
        otherwise
            error('CostFunction:UnknownCost', 'Unknown cost "%s".', NN.Cost);
    end
end

function prediction = evaluateNetwork(data, NN)
    switch NN.NetworkType
        case 'ANN'
            prediction = ANN(data, NN);
        case 'ResNet'
            prediction = ResNet(data, NN);
        otherwise
            error('CostFunction:UnsupportedNetworkType', ...
                'NetworkType must be ANN or ResNet.');
    end
end

function weightedValue = applySampleWeight(value, label, NN)
    if ~isfield(NN, 'WeightedFlag') || NN.WeightedFlag == 0
        weightedValue = value;
        return
    end

    if size(label, 2) == NN.numOfData
        sampleWeight = NN.Weighted;
    else
        sampleWeight = NN.SampleWeight;
    end
    weightedValue = sampleWeight .* value;
end
