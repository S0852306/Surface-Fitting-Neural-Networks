clear; clc; close all;

%% ------------------------------------------------------------
% 1) Generate data / label for fitting
%% ------------------------------------------------------------
n = 20;
x = linspace(-2, 2, n);
y = linspace(-2, 2, n);

[xGrid, yGrid] = meshgrid(x, y);       % (n x n)
radiusSquared = xGrid.^2 + yGrid.^2;

% data: (2 x n^2)
data = [xGrid(:)'; yGrid(:)'];

% label: (2 x n^2)
label1 = log(1 + (xGrid - 4/3).^2 + 3*(xGrid + yGrid - xGrid.^3).^2);
label2 = exp(-radiusSquared/2) .* cos(2*radiusSquared);
label  = [label1(:)'; label2(:)'];

%% ------------------------------------------------------------
% 2) Network setup + training
%% ------------------------------------------------------------
model.Cost = 'MSE';

layerStruct = [2, 10, 10, 10, 2];
model = Initialization(layerStruct, model);

option.MaxIteration = 600;
model = OptimizationSolver(data, label, model, option);

%% ------------------------------------------------------------
% 3) Validation (report + prediction)
%% ------------------------------------------------------------
report = FittingReport(data, label, model);
prediction = report.prediction;         % (2 x n^2)

%% ------------------------------------------------------------
% 4) Visualization (both outputs)
%% ------------------------------------------------------------
prediction = report.prediction;   % (2 x n^2)

fit1 = reshape(prediction(1,:), n, n);
fit2 = reshape(prediction(2,:), n, n);

figure;

% =========================
% Output 1
% =========================
subplot(1,2,1);
hold on; grid on;

scatter3(data(1,:), data(2,:), label(1,:), ...
    18, 'k', 'filled');

surf(xGrid, yGrid, fit1, ...
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

surf(xGrid, yGrid, fit2, ...
    'EdgeColor','none', ...
    'FaceAlpha',0.85);

xlabel('x'); ylabel('y'); zlabel('Output 2');
title('Neural Network Fit - Output 2');
legend({'Data','Fitted Surface'}, 'Location','best');
view(3);
