clear; clc; close all; % path automatically configured by startup or previous run
x = linspace(-2, 2, 20); y = x;
[X, Y] = meshgrid(x, y); U = X.^2 + Y.^2; Z = exp(-0.5*U).*cos(2*U);
data = [X(:), Y(:)].'; label = Z(:).';

NN = NeuralFit(data, label, [2, 1]); 
Prediction = NN.Evaluate(data);
PerformanceMetric = NN.Report;

figure();
surf(X, Y, Z); hold on; scatter3(data(1, :), data(2, :), Prediction)