clear; clc; close all;

%% ============================================================
%  Benchmark: Original Wavelet vs Learnable Spline Activation
%  Everything identical except the activation function.
%% ============================================================

%% 1) Generate data / label
n = 20;
x = linspace(-2, 2, n);
y = linspace(-2, 2, n);

[xGrid, yGrid] = meshgrid(x, y);
radiusSquared = xGrid.^2 + yGrid.^2;

data = [xGrid(:)'; yGrid(:)'];   % (2 x n^2)

label1 = log(1 + (xGrid - 4/3).^2 + 3*(xGrid + yGrid - xGrid.^3).^2);
label2 = exp(-radiusSquared/2) .* cos(2*radiusSquared);
label  = [label1(:)'; label2(:)'];  % (2 x n^2)

%% 2) Shared settings
inputDimension  = 2;
outputDimension = 2;
layerStruct = [inputDimension, 10, 16, 8, 10, outputDimension];

adamOption = struct();
adamOption.Solver       = 'ADAM';
adamOption.s0           = 1e-3;
adamOption.MaxIteration = 250;
adamOption.BatchSize    = 100;
adamOption.storeHistory = true;

bfgsOption = struct();
bfgsOption.Solver       = 'BFGS';
bfgsOption.MaxIteration = 500;
bfgsOption.storeHistory = true;

%% ============================================================
%  A) Original: Wavelet activation (custom handle)
%% ============================================================
fprintf('\n========== Training: Wavelet (original) ==========\n');

waveModel = struct();
waveModel.Cost             = 'MSE';
waveModel.NetworkType      = 'ResNet';
waveModel.InputAutoScaling = 'on';
waveModel.LabelAutoScaling = 'on';

waveAct = @(z) (1 - z.^2) .* exp(-0.5 * z.^2);
waveDer = @(z,a) exp(-0.5 * z.^2) .* (z.^3 - 3*z);
waveModel.active         = waveAct;
waveModel.activeDerivate = waveDer;

waveModel = Initialization(layerStruct, waveModel);

tWave = tic;
waveModel = OptimizationSolver(data, label, waveModel, adamOption);
waveModel = OptimizationSolver(data, label, waveModel, bfgsOption);
tWave = toc(tWave);

waveReport = FittingReport(data, label, waveModel);

%% ============================================================
%  B) Spline activation (learnable linear spline)
%% ============================================================
fprintf('\n========== Training: Learnable Spline ==========\n');

splineModel = struct();
splineModel.Cost             = 'MSE';
splineModel.NetworkType      = 'ResNet';
splineModel.InputAutoScaling = 'on';
splineModel.LabelAutoScaling = 'on';

splineModel.ActivationFunction = 'BSpline';
splineModel.bspline.order     = 4;           % cubic
splineModel.bspline.numGrid   = 8;
splineModel.bspline.gridRange = [-4 4];
splineModel.bspline.initShape = 'Wavelet';

splineModel = Initialization(layerStruct, splineModel);

tSpline = tic;
splineModel = OptimizationSolver(data, label, splineModel, adamOption);
splineModel = OptimizationSolver(data, label, splineModel, bfgsOption);
tSpline = toc(tSpline);

splineReport = FittingReport(data, label, splineModel);

%% ============================================================
%  3) Compare results
%% ============================================================
wavePrediction   = waveModel.Evaluate(data);
splinePrediction = splineModel.Evaluate(data);

waveMae   = mean(abs(label - wavePrediction), 'all');
splineMae = mean(abs(label - splinePrediction), 'all');

fprintf('\n==================== Summary ====================\n');
fprintf('  %-20s  %12s  %12s\n', '', 'Wavelet', 'Spline');
fprintf('  %-20s  %12.6e  %12.6e\n', 'Final Cost', ...
    waveModel.OptimizationHistory(end), splineModel.OptimizationHistory(end));
fprintf('  %-20s  %12.6e  %12.6e\n', 'MAE', waveMae, splineMae);
fprintf('  %-20s  %10.2f s   %10.2f s\n', 'Training Time', tWave, tSpline);
fprintf('  %-20s  %12d  %12d\n', 'Total Parameters', ...
    waveModel.numOfParameters, splineModel.numOfParameters);
fprintf('=================================================\n');

%% ============================================================
%  4) Training curves (same axes)
%% ============================================================
figure('Name','Training Curve Comparison');
hold on; grid on;
plot(log10(waveModel.OptimizationHistory),   '-o', ...
    'LineWidth', 1.3, 'MarkerIndices', 1:50:numel(waveModel.OptimizationHistory), ...
    'MarkerSize', 5);
plot(log10(splineModel.OptimizationHistory), '-s', ...
    'LineWidth', 1.3, 'MarkerIndices', 1:50:numel(splineModel.OptimizationHistory), ...
    'MarkerSize', 5);
xlabel('Iteration');
ylabel('log_{10}(Cost)');
title('Optimization History');
legend({'Wavelet (original)', 'Learnable Spline'}, 'Location', 'best');

%% ============================================================
%  5) Surface comparison
%% ============================================================
waveFit1   = reshape(wavePrediction(1,:),   n, n);
waveFit2   = reshape(wavePrediction(2,:),   n, n);
splineFit1 = reshape(splinePrediction(1,:), n, n);
splineFit2 = reshape(splinePrediction(2,:), n, n);

figure('Name','Output 1 Comparison');

subplot(1,2,1); hold on; grid on;
scatter3(data(1,:), data(2,:), label(1,:), 18, 'k', 'filled');
surf(xGrid, yGrid, waveFit1, 'EdgeColor','none', 'FaceAlpha',0.85);
title('Wavelet - Output 1');
xlabel('x'); ylabel('y'); zlabel('y1');
legend({'Data','Fit'}, 'Location','best'); view(3);

subplot(1,2,2); hold on; grid on;
scatter3(data(1,:), data(2,:), label(1,:), 18, 'k', 'filled');
surf(xGrid, yGrid, splineFit1, 'EdgeColor','none', 'FaceAlpha',0.85);
title('Spline - Output 1');
xlabel('x'); ylabel('y'); zlabel('y1');
legend({'Data','Fit'}, 'Location','best'); view(3);

figure('Name','Output 2 Comparison');

subplot(1,2,1); hold on; grid on;
scatter3(data(1,:), data(2,:), label(2,:), 18, 'k', 'filled');
surf(xGrid, yGrid, waveFit2, 'EdgeColor','none', 'FaceAlpha',0.85);
title('Wavelet - Output 2');
xlabel('x'); ylabel('y'); zlabel('y2');
legend({'Data','Fit'}, 'Location','best'); view(3);

subplot(1,2,2); hold on; grid on;
scatter3(data(1,:), data(2,:), label(2,:), 18, 'k', 'filled');
surf(xGrid, yGrid, splineFit2, 'EdgeColor','none', 'FaceAlpha',0.85);
title('Spline - Output 2');
xlabel('x'); ylabel('y'); zlabel('y2');
legend({'Data','Fit'}, 'Location','best'); view(3);

%% ============================================================
%  6) Visualize learned spline shapes
%% ============================================================
figure('Name','Learned B-Spline Shapes');
numHidden = splineModel.depth - 1;
zPlot = linspace(-5, 5, 500)';
for i = 1:numHidden
    subplot(1, numHidden, i); hold on; grid on;
    splineValue = BSplineActivation(zPlot, splineModel.splineCoeff{i}, splineModel.bsplineGrid);
    plot(zPlot, splineValue, '-', 'LineWidth', 1.5);
    xlabel('z'); ylabel('\sigma(z)');
    title(sprintf('Layer %d', i));
end
sgtitle('Learned B-Spline Activation Shapes');
