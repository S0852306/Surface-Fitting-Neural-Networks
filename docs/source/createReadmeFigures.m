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

staleFiles = {'high-accuracy-curve-fitting.png', 'surface-fitting-2d.png'};
for k = 1:numel(staleFiles)
    stalePath = fullfile(assetDir, staleFiles{k});
    if isfile(stalePath)
        delete(stalePath);
    end
end

%% Sharp 1D curve fitting
x = linspace(-3, 3, 220);
y = tanh(18*sin(5*x)) + 0.08*cos(17*x);

bsplineModel = NeuralFit(x, y, [1, 1], ...
    'stage1Iterations', 80, 'stage2Iterations', 500);
yHat = bsplineModel.Evaluate(x);

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 900 420]);
plot(x, y, 'k.', 'MarkerSize', 7); hold on;
plot(x, yHat, 'r-', 'LineWidth', 1.8);
grid on;
xlabel('x'); ylabel('y');
title(sprintf('Sharp square-wave-like 1D fit, MAE %.2e', bsplineModel.MeanAbsoluteError));
legend('Data', 'NeuralFit B-spline fit', 'Location', 'best');
exportgraphics(fig, fullfile(assetDir, 'sharp-square-wave-curve-fitting.png'), 'Resolution', 180);
close(fig);

%% Multi-output 2D surface fitting
n = 18;
x = linspace(-2, 2, n);
y = x;
[xGrid, yGrid] = meshgrid(x, y);
radiusSquared = xGrid.^2 + yGrid.^2;

output1 = log(1 + (xGrid - 4/3).^2 + 3*(xGrid + yGrid - xGrid.^3).^2);
output2 = exp(-radiusSquared/2).*cos(2*radiusSquared);
data = [xGrid(:), yGrid(:)].';
label = [output1(:).'; output2(:).'];

multiModel = NeuralFit(data, label, [2, 2], ...
    'stage1Iterations', 40, 'stage2Iterations', 220);
prediction = multiModel.Evaluate(data);
output1Hat = reshape(prediction(1,:), n, n);
output2Hat = reshape(prediction(2,:), n, n);

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1100 430]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
fitPlot = surf(xGrid, yGrid, output1Hat, 'EdgeColor', 'none'); hold on;
dataPlot = scatter3(data(1,:), data(2,:), label(1,:), 9, 'k', 'filled');
view(35, 24); grid on;
xlabel('x'); ylabel('y'); zlabel('y1');
title('NN fit - out1');
legend([dataPlot, fitPlot], {'Data', 'Fitting'}, 'Location', 'best');

nexttile;
fitPlot = surf(xGrid, yGrid, output2Hat, 'EdgeColor', 'none'); hold on;
dataPlot = scatter3(data(1,:), data(2,:), label(2,:), 9, 'k', 'filled');
view(35, 24); grid on;
xlabel('x'); ylabel('y'); zlabel('y2');
title('NN fit - out2');
legend([dataPlot, fitPlot], {'Data', 'Fitting'}, 'Location', 'best');

exportgraphics(fig, fullfile(assetDir, 'multi-output-surface-fitting.png'), 'Resolution', 180);
close(fig);

fprintf('README figures written to %s\n', assetDir);
