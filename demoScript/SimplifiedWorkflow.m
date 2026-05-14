clear; clc; close all;

%% ------------------------------------------------------------
% 1) Generate data / label for fitting
%% ------------------------------------------------------------
n = 20;
x = linspace(-2, 2, n);
y = linspace(-2, 2, n);

[X, Y] = meshgrid(x, y);               % (n x n)
U = X.^2 + Y.^2;

% data: (2 x n^2)
data = [X(:)'; Y(:)'];

% label: (2 x n^2)
label1 = log(1 + (X - 4/3).^2 + 3*(X + Y - X.^3).^2);
label2 = exp(-U/2) .* cos(2*U);
label  = [label1(:)'; label2(:)'];

%% ------------------------------------------------------------
% 2) Network setup + training
%% ------------------------------------------------------------
NN.Cost = 'MSE';

LayerStruct = [2, 10, 10, 10, 2];
NN = Initialization(LayerStruct, NN);

option.MaxIteration = 600;
NN = OptimizationSolver(data, label, NN, option);

%% ------------------------------------------------------------
% 3) Validation (report + prediction)
%% ------------------------------------------------------------
Report = FittingReport(data, label, NN);
Prediction = Report.Prediction;         % (2 x n^2)

%% ------------------------------------------------------------
% 4) Visualization (both outputs)
%% ------------------------------------------------------------
Prediction = Report.Prediction;   % (2 x n^2)

Zhat1 = reshape(Prediction(1,:), n, n);
Zhat2 = reshape(Prediction(2,:), n, n);

figure;

% =========================
% Output 1
% =========================
subplot(1,2,1);
hold on; grid on;

scatter3(data(1,:), data(2,:), label(1,:), ...
    18, 'k', 'filled');

surf(X, Y, Zhat1, ...
    'EdgeColor','none', ...
    'FaceAlpha',0.85);

xlabel('x'); ylabel('y'); zlabel('Output 1');
title('Neural Network Fit - Output 1');
legend({'Data','Fitted Surface'}, 'Location','best');
view(3);


% =========================
% Output 2
% =========================
subplot(1,2,2);
hold on; grid on;

scatter3(data(1,:), data(2,:), label(2,:), ...
    18, 'k', 'filled');

surf(X, Y, Zhat2, ...
    'EdgeColor','none', ...
    'FaceAlpha',0.85);

xlabel('x'); ylabel('y'); zlabel('Output 2');
title('Neural Network Fit - Output 2');
legend({'Data','Fitted Surface'}, 'Location','best');
view(3);