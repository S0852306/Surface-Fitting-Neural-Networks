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
LayerStruct = [1, 5, 5, 5, 1];

NN = struct();
NN.Cost = 'SSE';
NN.ActivationFunction = 'Wavelet';
NN = Initialization(LayerStruct, NN);

%% ------------------------------------------------------------
% 3) Training options (weighted SSE)
%% ------------------------------------------------------------
option = struct();
option.MaxIteration = 300;

% Weighting vector (same length as y)
w = ones(1, n);
w(300:end) = 10;                            % emphasize later samples
option.weighted = w;                        % comment this line to see difference

NN = OptimizationSolver(x, y, NN, option);

%% ------------------------------------------------------------
% 4) Report + prediction
%% ------------------------------------------------------------
Report = FittingReport(x, y, NN);
p = NN.Evaluate(x);

%% ------------------------------------------------------------
% 5) Visualization
%% ------------------------------------------------------------
figure; hold on; grid on;
plot(x, y, 'LineWidth', 1.5);
plot(x, p, 'LineWidth', 1.5);
xlabel('x'); ylabel('y');
title('1D Fit (Weighted SSE)');
legend({'Target','Prediction'}, 'Location','best');