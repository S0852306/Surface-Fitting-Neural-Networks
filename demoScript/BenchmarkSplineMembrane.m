clear; clc; close all;

%% ============================================================
%  MATLAB Logo (membrane) 2-D Fitting
%  Wavelet vs Learnable Spline
%% ============================================================

%% 1) Generate data / label from membrane
membraneSurface = membrane(1, 15);          % 31 x 31 surface
[numRows, numCols] = size(membraneSurface);
[xGrid, yGrid] = meshgrid(linspace(0, 1, numCols), linspace(0, 1, numRows));

data  = [xGrid(:)'; yGrid(:)'];   % (2 x 961)
label = membraneSurface(:)';      % (1 x 961)

%% 2) Shared settings
layerStruct = [2, 20, 32, 20, 1];

adamOption = struct();
adamOption.Solver       = 'ADAM';
adamOption.s0           = 1e-3;
adamOption.MaxIteration = 300;
adamOption.BatchSize    = 150;
adamOption.storeHistory = true;

bfgsOption = struct();
bfgsOption.Solver       = 'BFGS';
bfgsOption.MaxIteration = 600;
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

% waveAct = @(z) (1 - z.^2) .* exp(-0.5 * z.^2);
% waveDer = @(z,a) exp(-0.5 * z.^2) .* (z.^3 - 3*z);
% waveModel.active         = waveAct;
% waveModel.activeDerivate = waveDer;

waveModel = Initialization(layerStruct, waveModel);

tWave = tic;
waveModel = OptimizationSolver(data, label, waveModel, adamOption);
waveModel = OptimizationSolver(data, label, waveModel, bfgsOption);
tWave = toc(tWave);

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
splineModel.spline.initShape = 'Gaussian';

splineModel = Initialization(layerStruct, splineModel);

tSpline = tic;
splineModel = OptimizationSolver(data, label, splineModel, adamOption);
splineModel = OptimizationSolver(data, label, splineModel, bfgsOption);
tSpline = toc(tSpline);

%% ============================================================
%  3) FittingReport
%% ============================================================
fprintf('\n========== Wavelet report ==========\n');
waveReport = FittingReport(data, label, waveModel);

fprintf('\n========== Spline report ==========\n');
splineReport = FittingReport(data, label, splineModel);

%% ============================================================
%  4) Summary table
%% ============================================================
fprintf('\n============= Membrane Fitting Summary =============\n');
fprintf('  %-20s  %12s  %12s\n', '', 'Wavelet', 'Spline');
fprintf('  %-20s  %12.6e  %12.6e\n', 'Final Cost', ...
    waveModel.OptimizationHistory(end), splineModel.OptimizationHistory(end));
fprintf('  %-20s  %10.2f s   %10.2f s\n', 'Training Time', tWave, tSpline);
fprintf('  %-20s  %12d  %12d\n', 'Parameters', ...
    waveModel.numOfParameters, splineModel.numOfParameters);
fprintf('====================================================\n');

%% ============================================================
%  5) Training curves
%% ============================================================
figure('Name','Membrane: Training Curves');
hold on; grid on;
plot(log10(waveModel.OptimizationHistory),   '-o', ...
    'LineWidth', 1.3, 'MarkerIndices', 1:70:numel(waveModel.OptimizationHistory), ...
    'MarkerSize', 5);
plot(log10(splineModel.OptimizationHistory), '-s', ...
    'LineWidth', 1.3, 'MarkerIndices', 1:70:numel(splineModel.OptimizationHistory), ...
    'MarkerSize', 5);
xlabel('Iteration'); ylabel('log_{10}(Cost)');
title('Membrane Fitting: Optimization History');
legend({'Wavelet', 'Learnable Spline'}, 'Location', 'best');

%% ============================================================
%  6) Surface comparison on FINE grid (test generalization)
%% ============================================================
nFine = 100;
fineSurface = membrane(1, round((nFine-1)/2));  % produces nFine x nFine
[numFineRows, numFineCols] = size(fineSurface);
[xFineGrid, yFineGrid] = meshgrid(linspace(0, 1, numFineCols), linspace(0, 1, numFineRows));
dataFine = [xFineGrid(:)'; yFineGrid(:)'];

wavePredictionFine   = reshape(waveModel.Evaluate(dataFine),   numFineRows, numFineCols);
splinePredictionFine = reshape(splineModel.Evaluate(dataFine), numFineRows, numFineCols);

figure('Name','Membrane: Surface Comparison (fine grid)');

subplot(1,3,1);
surf(xFineGrid, yFineGrid, fineSurface, 'EdgeColor','none', 'FaceAlpha', 0.9);
title('Ground Truth'); xlabel('x'); ylabel('y'); zlabel('z');
colormap turbo; view(3); axis tight;

subplot(1,3,2);
surf(xFineGrid, yFineGrid, wavePredictionFine, 'EdgeColor','none', 'FaceAlpha', 0.9);
title('Wavelet'); xlabel('x'); ylabel('y'); zlabel('z');
colormap turbo; view(3); axis tight;

subplot(1,3,3);
surf(xFineGrid, yFineGrid, splinePredictionFine, 'EdgeColor','none', 'FaceAlpha', 0.9);
title('Spline'); xlabel('x'); ylabel('y'); zlabel('z');
colormap turbo; view(3); axis tight;

%% ============================================================
%  7) fitError heatmaps (fine grid)
%% ============================================================
waveErrorFine   = abs(fineSurface - wavePredictionFine);
splineErrorFine = abs(fineSurface - splinePredictionFine);
cmax = max(max(waveErrorFine(:)), max(splineErrorFine(:)));

fprintf('\n========== Fine-Grid Test fitError ==========\n');
fprintf('  Wavelet  MAE = %.6e\n', mean(waveErrorFine(:)));
fprintf('  Spline   MAE = %.6e\n', mean(splineErrorFine(:)));

figure('Name','Membrane: Pointwise fitError (fine grid)');

subplot(1,2,1);
imagesc(xFineGrid(1,:), yFineGrid(:,1), waveErrorFine);
set(gca,'YDir','normal'); caxis([0 cmax]); colorbar;
title('Wavelet |fitError|'); xlabel('x'); ylabel('y'); axis equal tight;

subplot(1,2,2);
imagesc(xFineGrid(1,:), yFineGrid(:,1), splineErrorFine);
set(gca,'YDir','normal'); caxis([0 cmax]); colorbar;
title('Spline |fitError|'); xlabel('x'); ylabel('y'); axis equal tight;

%% ============================================================
%  8) Learned spline shapes
%% ============================================================
figure('Name','Learned Spline Shapes');
numHidden = splineModel.depth - 1;
knots = splineModel.splineGrid.knots;
for i = 1:numHidden
    subplot(1, numHidden, i); hold on; grid on;
    plot(knots, splineModel.splineCoeff{i}, '-o', 'LineWidth', 1.5, 'MarkerSize', 3);
    xlabel('z'); ylabel('\sigma(z)');
    title(sprintf('Layer %d', i));
end
sgtitle('Learned Spline Activation per Layer');
