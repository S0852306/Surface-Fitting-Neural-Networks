%% Create README figures
% Regenerate the images used by README.md.

clear; clc; close all;

repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repoRoot, 'installScript'));
setupNeuralNetPath(struct('savePath', false, 'verbose', false));

assetDir = fullfile(repoRoot, 'docs', 'assets');
if ~isfolder(assetDir)
    mkdir(assetDir);
end

%% 1D high-accuracy curve fitting
x = linspace(-2, 2, 140);
y = abs(x) + 0.1*sin(8*x);

bsplineModel = NeuralFit(x, y, [1, 1], ...
    'stage1Iterations', 50, 'stage2Iterations', 300);
yHat = bsplineModel.Evaluate(x);

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 900 420]);
plot(x, y, 'k.', 'MarkerSize', 8); hold on;
plot(x, yHat, 'r-', 'LineWidth', 1.8);
grid on;
xlabel('x'); ylabel('y');
title(sprintf('High-accuracy 1D curve fitting, MAE %.2e', bsplineModel.MeanAbsoluteError));
legend('Data', 'NeuralFit B-spline fit', 'Location', 'best');
exportgraphics(fig, fullfile(assetDir, 'high-accuracy-curve-fitting.png'), 'Resolution', 180);
close(fig);

%% 2D nonlinear surface fitting
n = 24;
x = linspace(-2, 2, n);
y = x;
[X, Y] = meshgrid(x, y);
U = X.^2 + Y.^2;
Z = exp(-0.5*U).*cos(2*U);

data = [X(:), Y(:)].';
label = Z(:).';

surfaceModel = NeuralFit(data, label, [2, 1], ...
    'stage1Iterations', 30, 'stage2Iterations', 160);
prediction = reshape(surfaceModel.Evaluate(data), n, n);

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 980 420]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
surf(X, Y, Z, 'EdgeColor', 'none'); hold on;
scatter3(data(1,:), data(2,:), label, 8, 'k', 'filled');
view(35, 24); grid on;
xlabel('x'); ylabel('y'); zlabel('z');
title('Target surface');

nexttile;
surf(X, Y, prediction, 'EdgeColor', 'none'); hold on;
scatter3(data(1,:), data(2,:), label, 8, 'k', 'filled');
view(35, 24); grid on;
xlabel('x'); ylabel('y'); zlabel('z');
title('NeuralFit prediction');

exportgraphics(fig, fullfile(assetDir, 'surface-fitting-2d.png'), 'Resolution', 180);
close(fig);

%% Multi-output 2D surface fitting
n = 18;
x = linspace(-2, 2, n);
y = x;
[X, Y] = meshgrid(x, y);
U = X.^2 + Y.^2;

z1 = log(1 + (X - 4/3).^2 + 3*(X + Y - X.^3).^2);
z2 = exp(-U/2).*cos(2*U);
data = [X(:), Y(:)].';
label = [z1(:).'; z2(:).'];

multiModel = NeuralFit(data, label, [2, 2], ...
    'stage1Iterations', 40, 'stage2Iterations', 220);
prediction = multiModel.Evaluate(data);
z1Hat = reshape(prediction(1,:), n, n);
z2Hat = reshape(prediction(2,:), n, n);

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1100 430]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
surf(X, Y, z1Hat, 'EdgeColor', 'none'); hold on;
scatter3(data(1,:), data(2,:), label(1,:), 8, 'k', 'filled');
view(35, 24); grid on;
xlabel('x'); ylabel('y'); zlabel('y1');
title('Output 1 fit');

nexttile;
surf(X, Y, z2Hat, 'EdgeColor', 'none'); hold on;
scatter3(data(1,:), data(2,:), label(2,:), 8, 'k', 'filled');
view(35, 24); grid on;
xlabel('x'); ylabel('y'); zlabel('y2');
title('Output 2 fit');

exportgraphics(fig, fullfile(assetDir, 'multi-output-surface-fitting.png'), 'Resolution', 180);
close(fig);

fprintf('README figures written to %s\n', assetDir);
