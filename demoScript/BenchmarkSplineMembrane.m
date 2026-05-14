clear; clc; close all;

%% ============================================================
%  MATLAB Logo (membrane) 2-D Fitting
%  Wavelet vs Learnable Spline
%% ============================================================

%% 1) Generate data / label from membrane
L = membrane(1, 15);          % 31 x 31 surface
[nr, nc] = size(L);
[Xg, Yg] = meshgrid(linspace(0, 1, nc), linspace(0, 1, nr));

data  = [Xg(:)'; Yg(:)'];   % (2 x 961)
label = L(:)';               % (1 x 961)

%% 2) Shared settings
LayerStruct = [2, 20, 32, 20, 1];

optADAM = struct();
optADAM.Solver       = 'ADAM';
optADAM.s0           = 1e-3;
optADAM.MaxIteration = 300;
optADAM.BatchSize    = 150;
optADAM.storeHistory = true;

optBFGS = struct();
optBFGS.Solver       = 'BFGS';
optBFGS.MaxIteration = 600;
optBFGS.storeHistory = true;

%% ============================================================
%  A) Wavelet activation
%% ============================================================
fprintf('\n========== Training: Wavelet ==========\n');

NN_wave = struct();
NN_wave.Cost             = 'MSE';
NN_wave.NetworkType      = 'ResNet';
NN_wave.InputAutoScaling = 'on';
NN_wave.LabelAutoScaling = 'on';

% waveAct = @(z) (1 - z.^2) .* exp(-0.5 * z.^2);
% waveDer = @(z,a) exp(-0.5 * z.^2) .* (z.^3 - 3*z);
% NN_wave.active         = waveAct;
% NN_wave.activeDerivate = waveDer;

NN_wave = Initialization(LayerStruct, NN_wave);

tWave = tic;
NN_wave = OptimizationSolver(data, label, NN_wave, optADAM);
NN_wave = OptimizationSolver(data, label, NN_wave, optBFGS);
tWave = toc(tWave);

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
NN_spline.spline.initShape = 'Gaussian';

NN_spline = Initialization(LayerStruct, NN_spline);

tSpline = tic;
NN_spline = OptimizationSolver(data, label, NN_spline, optADAM);
NN_spline = OptimizationSolver(data, label, NN_spline, optBFGS);
tSpline = toc(tSpline);

%% ============================================================
%  3) FittingReport
%% ============================================================
fprintf('\n========== Wavelet Report ==========\n');
Report_wave = FittingReport(data, label, NN_wave);

fprintf('\n========== Spline Report ==========\n');
Report_spline = FittingReport(data, label, NN_spline);

%% ============================================================
%  4) Summary table
%% ============================================================
fprintf('\n============= Membrane Fitting Summary =============\n');
fprintf('  %-20s  %12s  %12s\n', '', 'Wavelet', 'Spline');
fprintf('  %-20s  %12.6e  %12.6e\n', 'Final Cost', ...
    NN_wave.OptimizationHistory(end), NN_spline.OptimizationHistory(end));
fprintf('  %-20s  %10.2f s   %10.2f s\n', 'Training Time', tWave, tSpline);
fprintf('  %-20s  %12d  %12d\n', 'Parameters', ...
    NN_wave.numOfParameters, NN_spline.numOfParameters);
fprintf('====================================================\n');

%% ============================================================
%  5) Training curves
%% ============================================================
figure('Name','Membrane: Training Curves');
hold on; grid on;
plot(log10(NN_wave.OptimizationHistory),   '-o', ...
    'LineWidth', 1.3, 'MarkerIndices', 1:70:numel(NN_wave.OptimizationHistory), ...
    'MarkerSize', 5);
plot(log10(NN_spline.OptimizationHistory), '-s', ...
    'LineWidth', 1.3, 'MarkerIndices', 1:70:numel(NN_spline.OptimizationHistory), ...
    'MarkerSize', 5);
xlabel('Iteration'); ylabel('log_{10}(Cost)');
title('Membrane Fitting: Optimization History');
legend({'Wavelet', 'Learnable Spline'}, 'Location', 'best');

%% ============================================================
%  6) Surface comparison on FINE grid (test generalization)
%% ============================================================
nFine = 100;
Lf = membrane(1, round((nFine-1)/2));  % produces nFine x nFine
[nrf, ncf] = size(Lf);
[Xf, Yf] = meshgrid(linspace(0, 1, ncf), linspace(0, 1, nrf));
dataFine = [Xf(:)'; Yf(:)'];

Pred_wave_f   = reshape(NN_wave.Evaluate(dataFine),   nrf, ncf);
Pred_spline_f = reshape(NN_spline.Evaluate(dataFine), nrf, ncf);

figure('Name','Membrane: Surface Comparison (fine grid)');

subplot(1,3,1);
surf(Xf, Yf, Lf, 'EdgeColor','none', 'FaceAlpha', 0.9);
title('Ground Truth'); xlabel('x'); ylabel('y'); zlabel('z');
colormap turbo; view(3); axis tight;

subplot(1,3,2);
surf(Xf, Yf, Pred_wave_f, 'EdgeColor','none', 'FaceAlpha', 0.9);
title('Wavelet'); xlabel('x'); ylabel('y'); zlabel('z');
colormap turbo; view(3); axis tight;

subplot(1,3,3);
surf(Xf, Yf, Pred_spline_f, 'EdgeColor','none', 'FaceAlpha', 0.9);
title('Spline'); xlabel('x'); ylabel('y'); zlabel('z');
colormap turbo; view(3); axis tight;

%% ============================================================
%  7) Error heatmaps (fine grid)
%% ============================================================
Err_wave_f   = abs(Lf - Pred_wave_f);
Err_spline_f = abs(Lf - Pred_spline_f);
cmax = max(max(Err_wave_f(:)), max(Err_spline_f(:)));

fprintf('\n========== Fine-Grid Test Error ==========\n');
fprintf('  Wavelet  MAE = %.6e\n', mean(Err_wave_f(:)));
fprintf('  Spline   MAE = %.6e\n', mean(Err_spline_f(:)));

figure('Name','Membrane: Pointwise Error (fine grid)');

subplot(1,2,1);
imagesc(Xf(1,:), Yf(:,1), Err_wave_f);
set(gca,'YDir','normal'); caxis([0 cmax]); colorbar;
title('Wavelet |error|'); xlabel('x'); ylabel('y'); axis equal tight;

subplot(1,2,2);
imagesc(Xf(1,:), Yf(:,1), Err_spline_f);
set(gca,'YDir','normal'); caxis([0 cmax]); colorbar;
title('Spline |error|'); xlabel('x'); ylabel('y'); axis equal tight;

%% ============================================================
%  8) Learned spline shapes
%% ============================================================
figure('Name','Learned Spline Shapes');
numHidden = NN_spline.depth - 1;
knots = NN_spline.splineGrid.knots;
for i = 1:numHidden
    subplot(1, numHidden, i); hold on; grid on;
    plot(knots, NN_spline.splineCoeff{i}, '-o', 'LineWidth', 1.5, 'MarkerSize', 3);
    xlabel('z'); ylabel('\sigma(z)');
    title(sprintf('Layer %d', i));
end
sgtitle('Learned Spline Activation per Layer');
