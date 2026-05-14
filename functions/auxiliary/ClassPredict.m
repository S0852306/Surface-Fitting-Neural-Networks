function Class = ClassPredict(data,NN)
%CLASSPREDICT Return class index of network output.
%   Class = ClassPredict(data,NN) evaluates the network and returns the index
%   of the maximum probability for each sample.

Probability = NN.Evaluate(data);
[~,Class] = max(Probability);
end