clear; clc; close all;
if ~exist('NeuralFit','file')
    run('installScript/installNeuralNetsPack.m');
end

x = linspace(-2, 2, 80);
y = abs(x) + 0.1*sin(8*x);

model = NeuralFit(x, y, [1, 1]);
prediction = model.Evaluate(x);

figure();
plot(x, y, 'k.', x, prediction, 'r-', 'LineWidth', 1.5);
grid on;
legend('Data', 'NeuralFit default: B-spline basis', 'Location', 'best');
title('Sharp / nonsmooth fitting: default B-spline basis');
