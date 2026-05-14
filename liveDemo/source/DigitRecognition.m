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
addpath(fullfile(packRoot, 'installScript'));
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

model.Cost = 'Entropy';
model.ActivationFunction = 'Gaussian';
model = Initialization([2, 12, 2], model);

option.Solver = 'ADAM';
option.MaxIteration = 80;
option.BatchSize = 40;
model = OptimizationSolver(data, label, model, option);

predictedClass = model.Predict(data);
hold on;
scatter(data(1, predictedClass == 1), data(2, predictedClass == 1), 24, 'filled');
scatter(data(1, predictedClass == 2), data(2, predictedClass == 2), 24, 'filled');
axis equal; grid on;
legend('Class 1', 'Class 2', 'Location', 'best');
title(sprintf('Training accuracy: %.1f%%', model.Accuracy));
