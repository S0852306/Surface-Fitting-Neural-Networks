clear; clc; close all;

%% ------------------------------------------------------------
% 1) Generate 1D regression data
%% ------------------------------------------------------------
n = 1000;
x = linspace(0, 5, n);                      % 1 x n
y = exp(-0.5 * x) .* cos(8 * x);            % 1 x n

%% ------------------------------------------------------------
% 2) Network setup
%% ------------------------------------------------------------
layerStruct = [1, 5, 5, 5, 1];

model = struct();
model.Cost = 'SSE';
model.ActivationFunction = 'Wavelet';
model = Initialization(layerStruct, model);

%% ------------------------------------------------------------
% 3) Training options (weighted SSE)
%% ------------------------------------------------------------
option = struct();
option.MaxIteration = 300;

% Weighting vector (same length as y)
sampleWeight = ones(1, n);
sampleWeight(300:end) = 10;                 % emphasize later samples
option.weighted = sampleWeight;             % comment this line to see difference

model = OptimizationSolver(x, y, model, option);

%% ------------------------------------------------------------
% 4) report + prediction
%% ------------------------------------------------------------
report = FittingReport(x, y, model);
prediction = model.Evaluate(x);

%% ------------------------------------------------------------
% 5) Visualization
%% ------------------------------------------------------------
figure; hold on; grid on;
plot(x, y, 'LineWidth', 1.5);
plot(x, prediction, 'LineWidth', 1.5);
xlabel('x'); ylabel('y');
title('1D Fit (Weighted SSE)');
legend({'Target','prediction'}, 'Location','best');
