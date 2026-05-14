clear; clc; close all;
addpath(genpath('functions'));

x = linspace(-3, 3, 160);
y = sign(sin(3*x)) + 0.15*cos(11*x);

bsplineModel = NeuralFit(x, y, [1, 1]);
gaussianModel = NeuralFit(x, y, [1, 1], 'basis', 'fixed');

yBspline = bsplineModel.Evaluate(x);
yGaussian = gaussianModel.Evaluate(x);

figure();
plot(x, y, 'k.', 'DisplayName', 'Data'); hold on;
plot(x, yBspline, 'r-', 'LineWidth', 1.5, ...
    'DisplayName', 'Default: B-spline basis');
plot(x, yGaussian, 'b--', 'LineWidth', 1.5, ...
    'DisplayName', 'Fixed: Gaussian basis');
grid on;
legend('Location', 'best');
title('Use B-spline for sharp / nonsmooth data; fixed Gaussian for smoother data');
