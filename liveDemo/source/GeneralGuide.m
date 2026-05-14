%% NeuralNetsPack quick start
% Default fitting uses a learnable B-spline basis.
% Use it when the target is sharp or nonsmooth.

clear; clc; close all;
thisDir = fileparts(mfilename('fullpath'));
[~, thisFolder] = fileparts(thisDir);
if strcmp(thisFolder, 'source')
    packRoot = fileparts(fileparts(thisDir));
else
    packRoot = fileparts(thisDir);
end
addpath(packRoot);
addpath(fullfile(packRoot, 'installScript'));
setupNeuralNetPath(struct('savePath', false, 'verbose', false));

x = linspace(-2, 2, 80);
y = abs(x) + 0.1*sin(8*x);

model = NeuralFit(x, y, [1, 1]);
yHat = model.Evaluate(x);

plot(x, y, 'k.', x, yHat, 'r-', 'LineWidth', 1.5);
grid on;
legend('Data', 'B-spline fit', 'Location', 'best');
title('Default: learnable B-spline basis');

%% Fixed basis for smooth data
% Gaussian is the fixed-basis default alias.

fixedModel = NeuralFit(x, y, [1, 1], 'basis', 'fixed', ...
    'stage1Iterations', 20, 'stage2Iterations', 80);
fixedHat = fixedModel.Evaluate(x);

plot(x, y, 'k.', x, fixedHat, 'b-', 'LineWidth', 1.5);
grid on;
legend('Data', 'Fixed Gaussian fit', 'Location', 'best');
title("Fixed basis: basis='fixed' or basis='Gaussian'");
