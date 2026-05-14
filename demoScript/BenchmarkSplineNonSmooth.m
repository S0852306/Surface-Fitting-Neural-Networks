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
[X, Y] = meshgrid(x, y);

data = [X(:)'; Y(:)'];   % (2 x n^2)

% ---- Output 1: abs-based ridge + step discontinuity ----
%   |x-y|  creates a V-shaped ridge along the diagonal
%   step at x=0 introduces a true discontinuity
label1 = abs(X - Y) + 0.8 * (X > 0) .* sin(3*Y);

% ---- Output 2: cone + sawtooth ----
%   sqrt(x^2+y^2) is a cone with a cusp at the origin
%   mod(.,.) adds sawtooth ripples (piecewise linear, non-smooth)
R = sqrt(X.^2 + Y.^2);
label2 = R + 0.5 * mod(3*X + 2*Y, 1);

label = [label1(:)'; label2(:)'];  % (2 x n^2)

%% 2) Shared settings
InputDimension  = 2;
OutputDimension = 2;
LayerStruct = [InputDimension, 16, 24, 16, 16, OutputDimension];

optADAM = struct();
optADAM.Solver       = 'ADAM';
optADAM.s0           = 1e-3;
optADAM.MaxIteration = 300;
optADAM.BatchSize    = 150;

optBFGS = struct();
optBFGS.Solver       = 'BFGS';
optBFGS.MaxIteration = 700;

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

Report_wave = FittingReport(data, label, NN_wave);

%% ============================================================
%  B) Learnable Spline activation
%% ============================================================
fprintf('\n========== Training: Learnable Spline ==========\n');

NN_spline = struct();
NN_spline.Cost             = 'MSE';
NN_spline.NetworkType      = 'ResNet';
NN_spline.InputAutoScaling = 'on';
NN_spline.LabelAutoScaling = 'on';

NN_spline.ActivationFunction = 'Spline';
NN_spline.spline.numGrid   = 32;
NN_spline.spline.gridRange = [-4 4];
NN_spline.spline.initShape = 'Wavelet';

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

fprintf('\n=========== Non-Smooth Benchmark Summary ===========\n');
fprintf('  %-20s  %12s  %12s\n', '', 'Wavelet', 'Spline');
fprintf('  %-20s  %12.6e  %12.6e\n', 'Final Cost', ...
    NN_wave.OptimizationHistory(end), NN_spline.OptimizationHistory(end));
fprintf('  %-20s  %12.6e  %12.6e\n', 'MAE', MAE_wave, MAE_spline);
fprintf('  %-20s  %10.2f s   %10.2f s\n', 'Training Time', tWave, tSpline);
fprintf('  %-20s  %12d  %12d\n', 'Total Parameters', ...
    NN_wave.numOfParameters, NN_spline.numOfParameters);
fprintf('====================================================\n');

%% ============================================================
%  4) Training curves
%% ============================================================
figure('Name','Non-Smooth: Training Curves');
hold on; grid on;
plot(log10(NN_wave.OptimizationHistory),   '-o', ...
    'LineWidth', 1.3, 'MarkerIndices', 1:80:numel(NN_wave.OptimizationHistory), ...
    'MarkerSize', 5);
plot(log10(NN_spline.OptimizationHistory), '-s', ...
    'LineWidth', 1.3, 'MarkerIndices', 1:80:numel(NN_spline.OptimizationHistory), ...
    'MarkerSize', 5);
xlabel('Iteration');
ylabel('log_{10}(Cost)');
title('Non-Smooth Target: Optimization History');
legend({'Wavelet', 'Learnable Spline'}, 'Location', 'best');

%% ============================================================
%  5) Surface comparison — Output 1 (ridge + step)
%% ============================================================
Zhat_wave1   = reshape(Pred_wave(1,:),   n, n);
Zhat_spline1 = reshape(Pred_spline(1,:), n, n);
Ztrue1       = reshape(label(1,:), n, n);

figure('Name','Output 1: Ridge + Step');

subplot(1,3,1); hold on; grid on;
surf(X, Y, Ztrue1, 'EdgeColor','none', 'FaceAlpha', 0.9);
title('Ground Truth'); xlabel('x'); ylabel('y'); zlabel('y_1'); view(3);

subplot(1,3,2); hold on; grid on;
surf(X, Y, Zhat_wave1, 'EdgeColor','none', 'FaceAlpha', 0.9);
title('Wavelet'); xlabel('x'); ylabel('y'); zlabel('y_1'); view(3);

subplot(1,3,3); hold on; grid on;
surf(X, Y, Zhat_spline1, 'EdgeColor','none', 'FaceAlpha', 0.9);
title('Spline'); xlabel('x'); ylabel('y'); zlabel('y_1'); view(3);

%% ============================================================
%  6) Surface comparison — Output 2 (cone + sawtooth)
%% ============================================================
Zhat_wave2   = reshape(Pred_wave(2,:),   n, n);
Zhat_spline2 = reshape(Pred_spline(2,:), n, n);
Ztrue2       = reshape(label(2,:), n, n);

figure('Name','Output 2: Cone + Sawtooth');

subplot(1,3,1); hold on; grid on;
surf(X, Y, Ztrue2, 'EdgeColor','none', 'FaceAlpha', 0.9);
title('Ground Truth'); xlabel('x'); ylabel('y'); zlabel('y_2'); view(3);

subplot(1,3,2); hold on; grid on;
surf(X, Y, Zhat_wave2, 'EdgeColor','none', 'FaceAlpha', 0.9);
title('Wavelet'); xlabel('x'); ylabel('y'); zlabel('y_2'); view(3);

subplot(1,3,3); hold on; grid on;
surf(X, Y, Zhat_spline2, 'EdgeColor','none', 'FaceAlpha', 0.9);
title('Spline'); xlabel('x'); ylabel('y'); zlabel('y_2'); view(3);

%% ============================================================
%  7) Pointwise error heatmap
%% ============================================================
Err_wave1   = reshape(abs(label(1,:) - Pred_wave(1,:)),   n, n);
Err_spline1 = reshape(abs(label(1,:) - Pred_spline(1,:)), n, n);
Err_wave2   = reshape(abs(label(2,:) - Pred_wave(2,:)),   n, n);
Err_spline2 = reshape(abs(label(2,:) - Pred_spline(2,:)), n, n);

cmax1 = max(max(Err_wave1(:)), max(Err_spline1(:)));
cmax2 = max(max(Err_wave2(:)), max(Err_spline2(:)));

figure('Name','Pointwise Absolute Error');

subplot(2,2,1);
imagesc(x, y, Err_wave1); set(gca,'YDir','normal');
caxis([0 cmax1]); colorbar; title('Wavelet |err| — Out 1');
xlabel('x'); ylabel('y');

subplot(2,2,2);
imagesc(x, y, Err_spline1); set(gca,'YDir','normal');
caxis([0 cmax1]); colorbar; title('Spline |err| — Out 1');
xlabel('x'); ylabel('y');

subplot(2,2,3);
imagesc(x, y, Err_wave2); set(gca,'YDir','normal');
caxis([0 cmax2]); colorbar; title('Wavelet |err| — Out 2');
xlabel('x'); ylabel('y');

subplot(2,2,4);
imagesc(x, y, Err_spline2); set(gca,'YDir','normal');
caxis([0 cmax2]); colorbar; title('Spline |err| — Out 2');
xlabel('x'); ylabel('y');

%% ============================================================
%  8) Learned spline shapes
%% ============================================================
zPlot = linspace(-5, 5, 500)';
figure('Name','Learned Spline Shapes (Non-Smooth Problem)');
numHidden = NN_spline.depth - 1;
for i = 1:numHidden
    subplot(1, numHidden, i); hold on; grid on;
    aSpline = SplineActivation(zPlot, NN_spline.splineCoeff{i}, NN_spline.splineGrid);
    aWave   = waveAct(zPlot);
    plot(zPlot, aWave,   '--', 'LineWidth', 1.2);
    plot(zPlot, aSpline, '-',  'LineWidth', 1.5);
    xlabel('z'); ylabel('\sigma(z)');
    title(sprintf('Layer %d', i));
    legend({'Wavelet (init)', 'Learned'}, 'Location', 'best');
end
sgtitle('Learned Activation Shapes');
