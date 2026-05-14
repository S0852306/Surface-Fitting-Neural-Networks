function vector = MatrixToVec(cells, NN)
%MATRIXTOVEC  Pack a parameter cell array into one column vector.

    totalLength = 0;
    for idx = 1:NN.depth
        totalLength = totalLength + numel(cells{idx});
    end

    vector = zeros(totalLength, 1);
    offset = 0;
    for idx = 1:NN.depth
        values = cells{idx}(:);
        count = numel(values);
        vector(offset + 1:offset + count) = values;
        offset = offset + count;
    end
end
