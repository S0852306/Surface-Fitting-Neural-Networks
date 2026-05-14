%% Training tips
% Sharp or nonsmooth target: start with default B-spline.
% Smooth target: try basis='fixed' or basis='Gaussian'.
% Bigger model is not always better; increase iterations first.

clear; clc; close all;
thisDir = fileparts(mfilename('fullpath'));
[~, thisFolder] = fileparts(thisDir);
if strcmp(thisFolder, 'source')
    packRoot = fileparts(fileparts(thisDir));
else
    packRoot = fileparts(thisDir);
end
addpath(packRoot);
setupNeuralNetPath(struct('savePath', false, 'verbose', false));

x = linspace(-2, 2, 120);
ySharp = sign(sin(4*x));
ySmooth = exp(-x.^2).*cos(2*x);

sharpModel = NeuralFit(x, ySharp, [1, 1], ...
    'stage1Iterations', 20, 'stage2Iterations', 80);
smoothModel = NeuralFit(x, ySmooth, [1, 1], 'basis', 'fixed', ...
    'stage1Iterations', 20, 'stage2Iterations', 80);

tiledlayout(1, 2);
nexttile;
plot(x, ySharp, 'k.', x, sharpModel.Evaluate(x), 'r-', 'LineWidth', 1.4);
grid on; title('Sharp: B-spline');

nexttile;
plot(x, ySmooth, 'k.', x, smoothModel.Evaluate(x), 'b-', 'LineWidth', 1.4);
grid on; title('Smooth: fixed Gaussian');
