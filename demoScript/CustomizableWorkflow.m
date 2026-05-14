clear; clc; close all;

%% ------------------------------------------------------------
% 1) Generate data / label (vectorized)
%% ------------------------------------------------------------
n = 20;
x = linspace(-2, 2, n);
y = linspace(-2, 2, n);

[X, Y] = meshgrid(x, y);
U = X.^2 + Y.^2;

data = [X(:)'; Y(:)'];   % (2 x n^2)

label1 = log(1 + (X - 4/3).^2 + 3*(X + Y - X.^3).^2);
label2 = exp(-U/2) .* cos(2*U);
label  = [label1(:)'; label2(:)'];  % (2 x n^2)

%% ------------------------------------------------------------
% 2) Custom Network setup (ResNet + Wavelet activation)
%% ------------------------------------------------------------
InputDimension  = 2;
OutputDimension = 2;
LayerStruct = [InputDimension, 10, 16, 8, 10, OutputDimension];

% ---- Wavelet activation: Mexican hat (Ricker) ----
% psi(z)  = (1 - z^2) * exp(-z^2/2)
% dpsi/dz = exp(-z^2/2) * (z^3 - 3z)
waveAct = @(z) (1 - z.^2) .* exp(-0.5 * z.^2);
waveDer = @(z,a) exp(-0.5 * z.^2) .* (z.^3 - 3*z);

NN.Cost = 'MSE';
NN.NetworkType = 'ResNet';

NN.InputAutoScaling = 'on';
NN.LabelAutoScaling = 'on';

NN.active = waveAct;
NN.activeDerivate = waveDer;

NN = Initialization(LayerStruct, NN);

%% ------------------------------------------------------------
% 3) Stage 1: First-order (ADAM)
%% ------------------------------------------------------------
option = struct();
option.Solver = 'ADAM';
option.s0 = 1e-3;
option.MaxIteration = 250;
option.BatchSize = 100;
option.storeHistory = true;

NN = OptimizationSolver(data, label, NN, option);

%% ------------------------------------------------------------
% 4) Stage 2: Quasi-Newton (BFGS)
%% ------------------------------------------------------------
option.Solver = 'BFGS';
option.MaxIteration = 500;

NN = OptimizationSolver(data, label, NN, option);

%% ------------------------------------------------------------
% 5) Training curve
%% ------------------------------------------------------------
figure; grid on;
plot(log10(NN.OptimizationHistory), 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('log10(Cost)');
title('Optimization History');

%% ------------------------------------------------------------
% 6) Validation
%% ------------------------------------------------------------
Prediction = NN.Evaluate(data);   % (2 x n^2)
Error = label - Prediction;
Report = FittingReport(data, label, NN);

%% ------------------------------------------------------------
% 7) Visualization (output 1 & 2)
%% ------------------------------------------------------------
Zhat1 = reshape(Prediction(1,:), n, n);
Zhat2 = reshape(Prediction(2,:), n, n);

figure;

subplot(1,2,1); hold on; grid on;
scatter3(data(1,:), data(2,:), label(1,:), 18, 'k', 'filled');
surf(X, Y, Zhat1, 'EdgeColor','none', 'FaceAlpha',0.85);
title('Custom Wavelet Activation NN - Output 1');
xlabel('x'); ylabel('y'); zlabel('y1');
legend({'Data','Fitting'}, 'Location','best');
view(3);

subplot(1,2,2); hold on; grid on;
scatter3(data(1,:), data(2,:), label(2,:), 18, 'k', 'filled');
surf(X, Y, Zhat2, 'EdgeColor','none', 'FaceAlpha',0.85);
title('Custom Wavelet Activation NN - Output 2');
xlabel('x'); ylabel('y'); zlabel('y2');
legend({'Data','Fitting'}, 'Location','best');
view(3);
