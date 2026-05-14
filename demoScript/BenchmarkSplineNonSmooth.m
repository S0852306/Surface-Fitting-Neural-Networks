clear; clc; close all;

%% ============================================================
%  Benchmark: Non-smooth target — Wavelet vs Learnable Spline
%  Target has sharp edges, kinks, and a discontinuous ridge to
%  stress-test the local adaptivity of spline activations.
%% ============================================================

%% 1) Generate data / label
n = 30;
x = linspace(-2, 2, n);
y = linspace(-2, 2, n);
[xGrid, yGrid] = meshgrid(x, y);

data = [xGrid(:)'; yGrid(:)'];   % (2 x n^2)

% ---- Output 1: abs-based ridge + step discontinuity ----
%   |x-y|  creates a V-shaped ridge along the diagonal
%   step at x=0 introduces a true discontinuity
label1 = abs(xGrid - yGrid) + 0.8 * (xGrid > 0) .* sin(3*yGrid);

% ---- Output 2: cone + sawtooth ----
%   sqrt(x^2+y^2) is a cone with a cusp at the origin
%   mod(.,.) adds sawtooth ripples (piecewise linear, non-smooth)
radius = sqrt(xGrid.^2 + yGrid.^2);
label2 = radius + 0.5 * mod(3*xGrid + 2*yGrid, 1);

label = [label1(:)'; label2(:)'];  % (2 x n^2)

%% 2) Shared settings
inputDimension  = 2;
outputDimension = 2;
layerStruct = [inputDimension, 16, 24, 16, 16, outputDimension];

adamOption = struct();
adamOption.Solver       = 'ADAM';
adamOption.s0           = 1e-3;
adamOption.MaxIteration = 300;
adamOption.BatchSize    = 150;
adamOption.storeHistory = true;

bfgsOption = struct();
bfgsOption.Solver       = 'BFGS';
bfgsOption.MaxIteration = 700;
bfgsOption.storeHistory = true;

%% ============================================================
%  A) Wavelet activation
%% ============================================================
fprintf('\n========== Training: Wavelet ==========\n');

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
%  B) Learnable Spline activation
%% ============================================================
fprintf('\n========== Training: Learnable Spline ==========\n');

splineModel = struct();
splineModel.Cost             = 'MSE';
splineModel.NetworkType      = 'ResNet';
splineModel.InputAutoScaling = 'on';
splineModel.LabelAutoScaling = 'on';

splineModel.ActivationFunction = 'Spline';
splineModel.spline.numGrid   = 32;
splineModel.spline.gridRange = [-4 4];
splineModel.spline.initShape = 'Wavelet';

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

fprintf('\n=========== Non-Smooth Benchmark Summary ===========\n');
fprintf('  %-20s  %12s  %12s\n', '', 'Wavelet', 'Spline');
fprintf('  %-20s  %12.6e  %12.6e\n', 'Final Cost', ...
    waveModel.OptimizationHistory(end), splineModel.OptimizationHistory(end));
fprintf('  %-20s  %12.6e  %12.6e\n', 'MAE', waveMae, splineMae);
fprintf('  %-20s  %10.2f s   %10.2f s\n', 'Training Time', tWave, tSpline);
fprintf('  %-20s  %12d  %12d\n', 'Total Parameters', ...
    waveModel.numOfParameters, splineModel.numOfParameters);
fprintf('====================================================\n');

%% ============================================================
%  4) Training curves
%% ============================================================
figure('Name','Non-Smooth: Training Curves');
hold on; grid on;
plot(log10(waveModel.OptimizationHistory),   '-o', ...
    'LineWidth', 1.3, 'MarkerIndices', 1:80:numel(waveModel.OptimizationHistory), ...
    'MarkerSize', 5);
plot(log10(splineModel.OptimizationHistory), '-s', ...
    'LineWidth', 1.3, 'MarkerIndices', 1:80:numel(splineModel.OptimizationHistory), ...
    'MarkerSize', 5);
xlabel('Iteration');
ylabel('log_{10}(Cost)');
title('Non-Smooth Target: Optimization History');
legend({'Wavelet', 'Learnable Spline'}, 'Location', 'best');

%% ============================================================
%  5) Surface comparison — Output 1 (ridge + step)
%% ============================================================
waveFit1   = reshape(wavePrediction(1,:),   n, n);
splineFit1 = reshape(splinePrediction(1,:), n, n);
trueSurface1       = reshape(label(1,:), n, n);

figure('Name','Output 1: Ridge + Step');

subplot(1,3,1); hold on; grid on;
surf(xGrid, yGrid, trueSurface1, 'EdgeColor','none', 'FaceAlpha', 0.9);
title('Ground Truth'); xlabel('x'); ylabel('y'); zlabel('y1'); view(3);

subplot(1,3,2); hold on; grid on;
surf(xGrid, yGrid, waveFit1, 'EdgeColor','none', 'FaceAlpha', 0.9);
title('Wavelet'); xlabel('x'); ylabel('y'); zlabel('y1'); view(3);

subplot(1,3,3); hold on; grid on;
surf(xGrid, yGrid, splineFit1, 'EdgeColor','none', 'FaceAlpha', 0.9);
title('Spline'); xlabel('x'); ylabel('y'); zlabel('y1'); view(3);

%% ============================================================
%  6) Surface comparison — Output 2 (cone + sawtooth)
%% ============================================================
waveFit2   = reshape(wavePrediction(2,:),   n, n);
splineFit2 = reshape(splinePrediction(2,:), n, n);
trueSurface2       = reshape(label(2,:), n, n);

figure('Name','Output 2: Cone + Sawtooth');

subplot(1,3,1); hold on; grid on;
surf(xGrid, yGrid, trueSurface2, 'EdgeColor','none', 'FaceAlpha', 0.9);
title('Ground Truth'); xlabel('x'); ylabel('y'); zlabel('y2'); view(3);

subplot(1,3,2); hold on; grid on;
surf(xGrid, yGrid, waveFit2, 'EdgeColor','none', 'FaceAlpha', 0.9);
title('Wavelet'); xlabel('x'); ylabel('y'); zlabel('y2'); view(3);

subplot(1,3,3); hold on; grid on;
surf(xGrid, yGrid, splineFit2, 'EdgeColor','none', 'FaceAlpha', 0.9);
title('Spline'); xlabel('x'); ylabel('y'); zlabel('y2'); view(3);

%% ============================================================
%  7) Pointwise fitError heatmap
%% ============================================================
waveError1   = reshape(abs(label(1,:) - wavePrediction(1,:)),   n, n);
splineError1 = reshape(abs(label(1,:) - splinePrediction(1,:)), n, n);
waveError2   = reshape(abs(label(2,:) - wavePrediction(2,:)),   n, n);
splineError2 = reshape(abs(label(2,:) - splinePrediction(2,:)), n, n);

cmax1 = max(max(waveError1(:)), max(splineError1(:)));
cmax2 = max(max(waveError2(:)), max(splineError2(:)));

figure('Name','Pointwise Absolute fitError');

subplot(2,2,1);
imagesc(x, y, waveError1); set(gca,'YDir','normal');
caxis([0 cmax1]); colorbar; title('Wavelet |err| — Out 1');
xlabel('x'); ylabel('y');

subplot(2,2,2);
imagesc(x, y, splineError1); set(gca,'YDir','normal');
caxis([0 cmax1]); colorbar; title('Spline |err| — Out 1');
xlabel('x'); ylabel('y');

subplot(2,2,3);
imagesc(x, y, waveError2); set(gca,'YDir','normal');
caxis([0 cmax2]); colorbar; title('Wavelet |err| — Out 2');
xlabel('x'); ylabel('y');

subplot(2,2,4);
imagesc(x, y, splineError2); set(gca,'YDir','normal');
caxis([0 cmax2]); colorbar; title('Spline |err| — Out 2');
xlabel('x'); ylabel('y');

%% ============================================================
%  8) Learned spline shapes
%% ============================================================
zPlot = linspace(-5, 5, 500)';
figure('Name','Learned Spline Shapes (Non-Smooth Problem)');
numHidden = splineModel.depth - 1;
for i = 1:numHidden
    subplot(1, numHidden, i); hold on; grid on;
    splineValue = SplineActivation(zPlot, splineModel.splineCoeff{i}, splineModel.splineGrid);
    waveValue   = waveAct(zPlot);
    plot(zPlot, waveValue,   '--', 'LineWidth', 1.2);
    plot(zPlot, splineValue, '-',  'LineWidth', 1.5);
    xlabel('z'); ylabel('\sigma(z)');
    title(sprintf('Layer %d', i));
    legend({'Wavelet (init)', 'Learned'}, 'Location', 'best');
end
sgtitle('Learned Activation Shapes');
