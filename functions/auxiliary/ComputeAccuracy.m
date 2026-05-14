function Accuracy = ComputeAccuracy(data,label,NN)
%COMPUTEACCURACY  Compute classification accuracy given network predictions.
%   Accuracy = ComputeAccuracy(data,label,NN) uses the network's Evaluate
%   method to predict probabilities, then compares the predicted index with
%   the true label index (assumed to be one-hot encoded).

Probability = NN.Evaluate(data);
[~,PredictIndex] = max(Probability);
LabelIndex = NN.HotToIndex(label);
Accuracy = 100 * mean(LabelIndex == PredictIndex);
end