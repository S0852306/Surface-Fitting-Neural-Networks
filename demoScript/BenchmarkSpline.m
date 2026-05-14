clear; clc; close all;

%% ============================================================
%  Benchmark: Original Wavelet vs Learnable Spline Activation
%  Everything identical except the activation function.
%% ============================================================

%% 1) Generate data / label
n = 20;
x = linspace(-2, 2, n);
y = linspace(-2, 2, n);

[X, Y] = meshgrid(x, y);
U = X.^2 + Y.^2;

data = [X(:)'; Y(:)'];   % (2 x n^2)

label1 = log(1 + (X - 4/3).^2 + 3*(X + Y - X.^3).^2);
label2 = exp(-U/2) .* cos(2*U);
label  = [label1(:)'; label2(:)'];  % (2 x n^2)

%% 2) Shared settings
InputDimension  = 2;
OutputDimension = 2;
LayerStruct = [InputDimension, 10, 16, 8, 10, OutputDimension];

optADAM = struct();
optADAM.Solver       = 'ADAM';
optADAM.s0           = 1e-3;
optADAM.MaxIteration = 250;
optADAM.BatchSize    = 100;
optADAM.storeHistory = true;

optBFGS = struct();
optBFGS.Solver       = 'BFGS';
optBFGS.MaxIteration = 500;
optBFGS.storeHistory = true;

%% ============================================================
%  A) Original: Wavelet activation (custom handle)
%% ============================================================
fprintf('\n========== Training: Wavelet (original) ==========\n');

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

Report_wave = FittingReport(data, label, NN_wave);

%% ============================================================
%  B) Spline activation (learnable linear spline)
%% ============================================================
fprintf('\n========== Training: Learnable Spline ==========\n');

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

Report_spline = FittingReport(data, label, NN_spline);

%% ============================================================
%  3) Compare results
%% ============================================================
Pred_wave   = NN_wave.Evaluate(data);
Pred_spline = NN_spline.Evaluate(data);

MAE_wave   = mean(abs(label - Pred_wave), 'all');
MAE_spline = mean(abs(label - Pred_spline), 'all');

fprintf('\n==================== Summary ====================\n');
fprintf('  %-20s  %12s  %12s\n', '', 'Wavelet', 'Spline');
fprintf('  %-20s  %12.6e  %12.6e\n', 'Final Cost', ...
    NN_wave.OptimizationHistory(end), NN_spline.OptimizationHistory(end));
fprintf('  %-20s  %12.6e  %12.6e\n', 'MAE', MAE_wave, MAE_spline);
fprintf('  %-20s  %10.2f s   %10.2f s\n', 'Training Time', tWave, tSpline);
fprintf('  %-20s  %12d  %12d\n', 'Total Parameters', ...
    NN_wave.numOfParameters, NN_spline.numOfParameters);
fprintf('=================================================\n');

%% ============================================================
%  4) Training curves (same axes)
%% ============================================================
figure('Name','Training Curve Comparison');
hold on; grid on;
plot(log10(NN_wave.OptimizationHistory),   '-o', ...
    'LineWidth', 1.3, 'MarkerIndices', 1:50:numel(NN_wave.OptimizationHistory), ...
    'MarkerSize', 5);
plot(log10(NN_spline.OptimizationHistory), '-s', ...
    'LineWidth', 1.3, 'MarkerIndices', 1:50:numel(NN_spline.OptimizationHistory), ...
    'MarkerSize', 5);
xlabel('Iteration');
ylabel('log_{10}(Cost)');
title('Optimization History');
legend({'Wavelet (original)', 'Learnable Spline'}, 'Location', 'best');

%% ============================================================
%  5) Surface comparison
%% ============================================================
Zhat_wave1   = reshape(Pred_wave(1,:),   n, n);
Zhat_wave2   = reshape(Pred_wave(2,:),   n, n);
Zhat_spline1 = reshape(Pred_spline(1,:), n, n);
Zhat_spline2 = reshape(Pred_spline(2,:), n, n);

figure('Name','Output 1 Comparison');

subplot(1,2,1); hold on; grid on;
scatter3(data(1,:), data(2,:), label(1,:), 18, 'k', 'filled');
surf(X, Y, Zhat_wave1, 'EdgeColor','none', 'FaceAlpha',0.85);
title('Wavelet - Output 1');
xlabel('x'); ylabel('y'); zlabel('y_1');
legend({'Data','Fit'}, 'Location','best'); view(3);

subplot(1,2,2); hold on; grid on;
scatter3(data(1,:), data(2,:), label(1,:), 18, 'k', 'filled');
surf(X, Y, Zhat_spline1, 'EdgeColor','none', 'FaceAlpha',0.85);
title('Spline - Output 1');
xlabel('x'); ylabel('y'); zlabel('y_1');
legend({'Data','Fit'}, 'Location','best'); view(3);

figure('Name','Output 2 Comparison');

subplot(1,2,1); hold on; grid on;
scatter3(data(1,:), data(2,:), label(2,:), 18, 'k', 'filled');
surf(X, Y, Zhat_wave2, 'EdgeColor','none', 'FaceAlpha',0.85);
title('Wavelet - Output 2');
xlabel('x'); ylabel('y'); zlabel('y_2');
legend({'Data','Fit'}, 'Location','best'); view(3);

subplot(1,2,2); hold on; grid on;
scatter3(data(1,:), data(2,:), label(2,:), 18, 'k', 'filled');
surf(X, Y, Zhat_spline2, 'EdgeColor','none', 'FaceAlpha',0.85);
title('Spline - Output 2');
xlabel('x'); ylabel('y'); zlabel('y_2');
legend({'Data','Fit'}, 'Location','best'); view(3);

%% ============================================================
%  6) Visualize learned spline shapes
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
