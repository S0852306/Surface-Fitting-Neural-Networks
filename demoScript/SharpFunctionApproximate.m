%% 2D Sharp-but-Smooth Function Fitting with Pointwise Error
% Target: a 2D function with sharp but smooth transitions
clear; clc; close all;

%% Generate sharp-but-smooth target on [-2,2]x[-2,2]
n = 50;
xv = linspace(-2, 2, n);
yv = linspace(-2, 2, n);
[X, Y] = meshgrid(xv, yv);

% Smooth but with sharp transitions
% --- Parameters ---
F = zeros(size(X));
num_pulses = 3;
centers = linspace(-1.5, 1.5, num_pulses);
width = 0.35; % width of each pulse
sharpness = 40; % higher = sharper edge
for i = 1:num_pulses
    for j = 1:num_pulses
        cx = centers(i);
        cy = centers(j);
        % 2D smooth square pulse (almost discrete)
        pulse = 0.8 * (1 ./ (1 + exp(-sharpness*(X-cx+width/2))) - 1 ./ (1 + exp(-sharpness*(X-cx-width/2)))) ...
                   .* (1 ./ (1 + exp(-sharpness*(Y-cy+width/2))) - 1 ./ (1 + exp(-sharpness*(Y-cy-width/2))));
        F = F + pulse;
    end
end

% Add small amplitude smooth variation
F = F + 0.15 * sin(2*X) .* sin(2*Y);

data  = [X(:), Y(:)].';   % 2 x D
label = F(:).';           % 1 x D

%% Fit with deep ResNet + BSpline activation (ReLU init shape) + LayerNorm
layers = [2, repmat(8,1,30), 1];  % 20 hidden layers, width 8

NN.Cost = 'MSE';
NN.ActivationFunction = 'Gaussian';
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
performance = FittingReport(data, label, NN);
%% Compute pointwise absolute error
errMap = reshape(abs(prediction - label), n, n);
predSurf = reshape(prediction, n, n);

%% Plot
figure('Color','w', 'Position', [80 80 1300 500]);

% --- Left: true surface ---
subplot(1,3,1);
surf(X, Y, F, 'EdgeColor', 'none');
xlabel('x'); ylabel('y'); zlabel('f');
title('Target (sharp-but-smooth)');
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

fprintf('Sharp-but-smooth 2D fit — MAE: %.4e, max|error|: %.4e\n', ...
    mean(errMap(:)), max(errMap(:)));
