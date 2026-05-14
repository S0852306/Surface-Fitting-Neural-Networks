%% Model in one page
% A network maps data x to prediction yHat.
% Training minimizes CostFunction(data, label, NN).

clear; clc;
thisDir = fileparts(mfilename('fullpath'));
[~, thisFolder] = fileparts(thisDir);
if strcmp(thisFolder, 'source')
    packRoot = fileparts(fileparts(thisDir));
else
    packRoot = fileparts(thisDir);
end
addpath(packRoot);
setupNeuralNetPath(struct('savePath', false, 'verbose', false));

%% Basis choices
% basis='bspline' learns the hidden nonlinearity.
% basis='fixed' uses Gaussian.
% Other fixed choices: Gaussian, tanh, ReLU, Wavelet.

x = linspace(-1, 1, 20);
y = x.^2;

bsplineModel = NeuralFit(x, y, [1, 1], ...
    'stage1Iterations', 5, 'stage2Iterations', 10);
gaussianModel = NeuralFit(x, y, [1, 1], 'basis', 'Gaussian', ...
    'stage1Iterations', 5, 'stage2Iterations', 10);

fprintf('B-spline basis: %s, order %d, grid %d\n', ...
    bsplineModel.ActivationFunction, bsplineModel.bspline.order, bsplineModel.bspline.numGrid);
fprintf('Fixed basis: %s\n', gaussianModel.ActivationFunction);
