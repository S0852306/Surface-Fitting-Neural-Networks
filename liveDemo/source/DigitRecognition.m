%% Compact classification demo
% This file keeps the classification workflow short.

clear; clc; close all;
thisDir = fileparts(mfilename('fullpath'));
[~, thisFolder] = fileparts(thisDir);
if strcmp(thisFolder, 'source')
    packRoot = fileparts(fileparts(thisDir));
else
    packRoot = fileparts(thisDir);
end
addpath(packRoot);
setupNeuralNetPath(struct('savePath', false, 'verbose', false));

rng(2);
n = 80;
x1 = randn(2, n)*0.25 + [-0.7; 0.0];
x2 = randn(2, n)*0.25 + [ 0.7; 0.0];
data = [x1, x2];

classId = [ones(1, n), 2*ones(1, n)];
label = zeros(2, 2*n);
label(1, classId == 1) = 1;
label(2, classId == 2) = 1;

NN.Cost = 'Entropy';
NN.ActivationFunction = 'Gaussian';
NN = Initialization([2, 12, 2], NN);

option.Solver = 'ADAM';
option.MaxIteration = 80;
option.BatchSize = 40;
NN = OptimizationSolver(data, label, NN, option);

predictedClass = NN.Predict(data);
gscatter(data(1,:), data(2,:), predictedClass);
axis equal; grid on;
title(sprintf('Training accuracy: %.1f%%', NN.Accuracy));
