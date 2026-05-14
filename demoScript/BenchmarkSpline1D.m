clear; clc; close all;

%% ============================================================
%  Benchmark 1-D: Square-wave-like + smooth perturbation
%  Wavelet vs Learnable B-Spline
%% ============================================================

%% 1) Generate data / label
n = 400;
[data, label] = generateTarget(linspace(-3, 3, n));

%% 2) Shared settings
LayerStruct = [1, 24, 32, 24, 1];

optADAM = struct();
optADAM.Solver       = 'ADAM';
optADAM.s0           = 1e-3;
optADAM.MaxIteration = 400;
optADAM.BatchSize    = 80;

optBFGS = struct();
optBFGS.Solver       = 'BFGS';
optBFGS.MaxIteration = 600;

%% ============================================================
%  A) Wavelet activation
%% ============================================================
fprintf('\n========== Training: Wavelet ==========\n');

NN_wave = struct();
NN_wave.Cost             = 'MSE';
NN_wave.NetworkType      = 'ResNet';
NN_wave.InputAutoScaling = 'on';
NN_wave.LabelAutoScaling = 'on';

waveAct = @(z) (1 - z.^2) .* exp(-0.5 * z.^2);
waveDer = @(z,a) exp(-0.5 * z.^2) .* (z.^3 - 3*z);
NN_wave.active         = waveAct;
NN_wave.activeDerivate = waveDer;

NN_wave = Initialization(LayerStruct, NN_wave);

tWave = tic;
NN_wave = OptimizationSolver(data, label, NN_wave, optADAM);
NN_wave = OptimizationSolver(data, label, NN_wave, optBFGS);
tWave = toc(tWave);

%% ============================================================
%  B) Learnable B-Spline activation
%% ============================================================
fprintf('\n========== Training: Learnable B-Spline ==========\n');

NN_spline = struct();
NN_spline.Cost             = 'MSE';
NN_spline.NetworkType      = 'ResNet';
NN_spline.InputAutoScaling = 'on';
NN_spline.LabelAutoScaling = 'on';

NN_spline.ActivationFunction = 'BSpline';
NN_spline.bspline.order     = 4;           % cubic
NN_spline.bspline.numGrid   = 8;
NN_spline.bspline.gridRange = [-4 4];
NN_spline.bspline.initShape = 'Wavelet';

NN_spline = Initialization(LayerStruct, NN_spline);

tSpline = tic;
NN_spline = OptimizationSolver(data, label, NN_spline, optADAM);
NN_spline = OptimizationSolver(data, label, NN_spline, optBFGS);
tSpline = toc(tSpline);

%% ============================================================
%  3) Evaluate
%% ============================================================
[xFine, yTrue] = generateTarget(linspace(-3, 3, 1000));

yWave   = NN_wave.Evaluate(xFine);
ySpline = NN_spline.Evaluate(xFine);

MAE_wave   = mean(abs(label - NN_wave.Evaluate(data)));
MAE_spline = mean(abs(label - NN_spline.Evaluate(data)));

fprintf('\n============ 1-D Non-Smooth Summary ============\n');
fprintf('  %-20s  %12s  %12s\n', '', 'Wavelet', 'B-Spline');
fprintf('  %-20s  %12.6e  %12.6e\n', 'Final Cost', ...
    NN_wave.OptimizationHistory(end), NN_spline.OptimizationHistory(end));
fprintf('  %-20s  %12.6e  %12.6e\n', 'MAE', MAE_wave, MAE_spline);
fprintf('  %-20s  %10.2f s   %10.2f s\n', 'Time', tWave, tSpline);
fprintf('  %-20s  %12d  %12d\n', 'Parameters', ...
    NN_wave.numOfParameters, NN_spline.numOfParameters);
fprintf('================================================\n');

%% ============================================================
%  4) Training curves
%% ============================================================
figure('Name','1-D Training Curves');
hold on; grid on;
plot(log10(NN_wave.OptimizationHistory),   '-o', ...
    'LineWidth', 1.3, 'MarkerIndices', 1:80:numel(NN_wave.OptimizationHistory), ...
    'MarkerSize', 5);
plot(log10(NN_spline.OptimizationHistory), '-s', ...
    'LineWidth', 1.3, 'MarkerIndices', 1:80:numel(NN_spline.OptimizationHistory), ...
    'MarkerSize', 5);
xlabel('Iteration');
ylabel('log_{10}(Cost)');
title('Square-Wave Fitting: Optimization History');
legend({'Wavelet', 'Learnable B-Spline'}, 'Location', 'best');

%% ============================================================
%  5) Fitting overlay
%% ============================================================
figure('Name','1-D Fitting Comparison');

subplot(2,1,1); hold on; grid on;
plot(xFine, yTrue, 'k-', 'LineWidth', 1.8);
plot(xFine, yWave, '--', 'LineWidth', 1.3);
plot(xFine, ySpline, '-.', 'LineWidth', 1.3);
xlabel('x'); ylabel('y');
title('Fitting Result');
legend({'Ground Truth', 'Wavelet', 'B-Spline'}, 'Location', 'best');

subplot(2,1,2); hold on; grid on;
plot(xFine, abs(yTrue - yWave),   '--', 'LineWidth', 1.2);
plot(xFine, abs(yTrue - ySpline), '-.', 'LineWidth', 1.2);
xlabel('x'); ylabel('|error|');
title('Pointwise Absolute Error');
legend({'Wavelet', 'B-Spline'}, 'Location', 'best');

%% ============================================================
%  6) Zoom near sharp transitions
%% ============================================================
figure('Name','Zoom: Sharp Edges');

edges = [0, 1, 2];
for ei = 1:3
    xc = edges(ei);
    mask = xFine > xc - 0.3 & xFine < xc + 0.3;

    subplot(1,3,ei); hold on; grid on;
    plot(xFine(mask), yTrue(mask),   'k-',  'LineWidth', 1.8);
    plot(xFine(mask), yWave(mask),   '--',  'LineWidth', 1.3);
    plot(xFine(mask), ySpline(mask), '-.',  'LineWidth', 1.3);
    xlabel('x'); ylabel('y');
    title(sprintf('Edge near x = %g', xc));
    legend({'Truth', 'Wavelet', 'B-Spline'}, 'Location', 'best');
end
sgtitle('Zoom at Sharp Transitions');

%% ============================================================
%  7) Learned B-spline shapes
%% ============================================================
figure('Name','Learned B-Spline Shapes');
numHidden = NN_spline.depth - 1;
zPlot = linspace(-5, 5, 500)';
for i = 1:numHidden
    subplot(1, numHidden, i); hold on; grid on;
    aBS = BSplineActivation(zPlot, NN_spline.splineCoeff{i}, NN_spline.bsplineGrid);
    plot(zPlot, aBS, '-', 'LineWidth', 1.5);
    xlabel('z'); ylabel('\sigma(z)');
    title(sprintf('Layer %d', i));
end
sgtitle('Learned B-Spline Activation Shapes');

%% ============================================================
%  Local function: single source of truth for the target
%% ============================================================
function [x, y] = generateTarget(x)
    k = 80;  % steepness of tanh steps
    sqwave      = tanh(k*x) - tanh(k*(x-1)) + tanh(k*(x-2)) - 2;
    smooth_part = 0.05 * sin(2*pi*x) + 0.05 * cos(5*pi*x);
    y = sqwave + smooth_part;
end
