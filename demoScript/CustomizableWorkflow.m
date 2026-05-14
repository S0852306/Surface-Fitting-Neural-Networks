clear; clc; close all;

%% ------------------------------------------------------------
% 1) Generate data / label (vectorized)
%% ------------------------------------------------------------
n = 20;
x = linspace(-2, 2, n);
y = linspace(-2, 2, n);

[xGrid, yGrid] = meshgrid(x, y);
radiusSquared = xGrid.^2 + yGrid.^2;

data = [xGrid(:)'; yGrid(:)'];   % (2 x n^2)

label1 = log(1 + (xGrid - 4/3).^2 + 3*(xGrid + yGrid - xGrid.^3).^2);
label2 = exp(-radiusSquared/2) .* cos(2*radiusSquared);
label  = [label1(:)'; label2(:)'];  % (2 x n^2)

%% ------------------------------------------------------------
% 2) Custom Network setup (ResNet + Wavelet activation)
%% ------------------------------------------------------------
inputDimension  = 2;
outputDimension = 2;
layerStruct = [inputDimension, 10, 16, 8, 10, outputDimension];

% ---- Wavelet activation: Mexican hat (Ricker) ----
% psi(z)  = (1 - z^2) * exp(-z^2/2)
% dpsi/dz = exp(-z^2/2) * (z^3 - 3z)
waveAct = @(z) (1 - z.^2) .* exp(-0.5 * z.^2);
waveDer = @(z,a) exp(-0.5 * z.^2) .* (z.^3 - 3*z);

model.Cost = 'MSE';
model.NetworkType = 'ResNet';

model.InputAutoScaling = 'on';
model.LabelAutoScaling = 'on';

model.active = waveAct;
model.activeDerivate = waveDer;

model = Initialization(layerStruct, model);

%% ------------------------------------------------------------
% 3) Stage 1: First-order (ADAM)
%% ------------------------------------------------------------
option = struct();
option.Solver = 'ADAM';
option.s0 = 1e-3;
option.MaxIteration = 250;
option.BatchSize = 100;
option.storeHistory = true;

model = OptimizationSolver(data, label, model, option);

%% ------------------------------------------------------------
% 4) Stage 2: Quasi-Newton (BFGS)
%% ------------------------------------------------------------
option.Solver = 'BFGS';
option.MaxIteration = 500;

model = OptimizationSolver(data, label, model, option);

%% ------------------------------------------------------------
% 5) Training curve
%% ------------------------------------------------------------
figure; grid on;
plot(log10(model.OptimizationHistory), 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('log10(Cost)');
title('Optimization History');

%% ------------------------------------------------------------
% 6) Validation
%% ------------------------------------------------------------
prediction = model.Evaluate(data);   % (2 x n^2)
fitError = label - prediction;
report = FittingReport(data, label, model);

%% ------------------------------------------------------------
% 7) Visualization (output 1 & 2)
%% ------------------------------------------------------------
fit1 = reshape(prediction(1,:), n, n);
fit2 = reshape(prediction(2,:), n, n);

figure;

subplot(1,2,1); hold on; grid on;
scatter3(data(1,:), data(2,:), label(1,:), 18, 'k', 'filled');
surf(xGrid, yGrid, fit1, 'EdgeColor','none', 'FaceAlpha',0.85);
title('Custom Wavelet Activation NN - Output 1');
xlabel('x'); ylabel('y'); zlabel('y1');
legend({'Data','Fitting'}, 'Location','best');
view(3);

subplot(1,2,2); hold on; grid on;
scatter3(data(1,:), data(2,:), label(2,:), 18, 'k', 'filled');
surf(xGrid, yGrid, fit2, 'EdgeColor','none', 'FaceAlpha',0.85);
title('Custom Wavelet Activation NN - Output 2');
xlabel('x'); ylabel('y'); zlabel('y2');
legend({'Data','Fitting'}, 'Location','best');
view(3);
