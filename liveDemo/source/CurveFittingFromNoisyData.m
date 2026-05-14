%% Curve fitting from noisy data
% Keep the interface small: data, label, dimensions, optional basis.

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

rng(1);
x = linspace(-3, 3, 120);
clean = sin(3*x) + 0.4*abs(x);
y = clean + 0.08*randn(size(x));

model = NeuralFit(x, y, [1, 1], ...
    'stage1Iterations', 30, 'stage2Iterations', 150);
yHat = model.Evaluate(x);

plot(x, y, 'k.', x, clean, 'Color', [0.6 0.6 0.6], 'LineWidth', 1.0); hold on;
plot(x, yHat, 'r-', 'LineWidth', 1.6);
grid on;
legend('Noisy data', 'Clean target', 'Fit', 'Location', 'best');
title('Noisy nonsmooth curve: default B-spline basis');
