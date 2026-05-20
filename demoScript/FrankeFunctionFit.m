%% 2D Piecewise-Constant Surface Fitting with Pointwise Error
% Target: a 2D function with multiple plateau regions and sharp jumps
clear; clc; close all;

%% Generate piecewise-constant target on [-2,2]x[-2,2]
n = 50;
xv = linspace(-2, 2, n);
yv = linspace(-2, 2, n);
[X, Y] = meshgrid(xv, yv);

% Multiple step regions (values in [-1, 1])
F = 0.6*sign(sin(2*X)) ...
  + 0.4*sign(cos(3*Y)) ...
  - 0.3*sign(X.*Y - 0.5) ...
  + 0.2*sign(X.^2 + Y.^2 - 2);

data  = [X(:), Y(:)].';   % 2 x D
label = F(:).';            % 1 x D

%% Fit with deep ResNet + BSpline activation (ReLU init shape) + LayerNorm
layers = [2, repmat(8,1,20), 1];  % 20 hidden layers, width 8

NN.Cost = 'MSE';
NN.ActivationFunction = 'Gaussian';
% NN.BSplineInitShape = 'ReLU';
NN.NetworkType = 'ResNet';
NN.LayerNorm = 'on';
NN = Initialization(layers, NN);

option.Solver = 'ADAM';
option.MaxIteration = 150;
option.BatchSize = 200;
NN = OptimizationSolver(data, label, NN, option);

option.Solver = 'BFGS';
option.MaxIteration = 450;
NN = OptimizationSolver(data, label, NN, option);

prediction = NN.Evaluate(data);

%% Compute pointwise absolute error
errMap = reshape(abs(prediction - label), n, n);
predSurf = reshape(prediction, n, n);

%% Plot
figure('Color','w', 'Position', [80 80 1300 500]);

% --- Left: true piecewise surface ---
subplot(1,3,1);
surf(X, Y, F, 'EdgeColor', 'none');
xlabel('x'); ylabel('y'); zlabel('f');
title('Target (piecewise-constant)');
view([-35 30]); colormap(gca, parula); colorbar;

% --- Middle: NN fit ---
subplot(1,3,2);
surf(X, Y, predSurf, 'EdgeColor', 'none');
xlabel('x'); ylabel('y'); zlabel('f');
title(sprintf('NN Fit (MAE = %.3e)', mean(errMap(:))));
view([-35 30]); colormap(gca, parula); colorbar;

% --- Right: pointwise error heatmap ---
subplot(1,3,3);
contourf(X, Y, errMap, 20, 'LineColor', 'none');
xlabel('x'); ylabel('y');
title('Pointwise |error|');
colorbar; colormap(gca, hot);
axis equal tight;

fprintf('Piecewise 2D fit — MAE: %.4e, max|error|: %.4e\n', ...
    mean(errMap(:)), max(errMap(:)));
