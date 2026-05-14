function cells = VecToMatrix(vector, NN)
%VECTOMATRIX  Unpack a vector into NN-shaped weight or bias cells.

    vector = vector(:);
    if numel(vector) == NN.numOfWeight
        cells = vectorToWeights(vector, NN);
    elseif numel(vector) == NN.numOfBias
        cells = vectorToBiases(vector, NN);
    else
        error('VecToMatrix:BadVectorLength', ...
            'Vector length must match NN.numOfWeight or NN.numOfBias.');
    end
end

function weights = vectorToWeights(vector, NN)
    weights = NN.weight;
    offset = 0;
    for idx = 1:NN.depth
        outDim = NN.LayerStruct(2, idx);
        inDim = NN.LayerStruct(1, idx);
        count = outDim * inDim;
        weights{idx} = reshape(vector(offset + 1:offset + count), outDim, inDim);
        offset = offset + count;
    end
end

function biases = vectorToBiases(vector, NN)
    biases = NN.bias;
    offset = 0;
    for idx = 1:NN.depth
        count = NN.LayerStruct(2, idx);
        biases{idx} = reshape(vector(offset + 1:offset + count), count, 1);
        offset = offset + count;
    end
end
