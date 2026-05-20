clear; clc; close all;

if ~exist('NeuralFit','file')
    run('installScript/installNeuralNetsPack.m');
end

x = linspace(-2, 2, 100);
y = abs(x) + 0.15*sin(10*x).*exp(-0.4*x.^2);

model = NeuralFit(x, y, [1, 1]);
prediction = model.Evaluate(x);

figure();
plot(x, y, 'k.', x, prediction, 'r-', 'LineWidth', 1.5);
grid on;
legend('Data', 'NeuralFit', 'Location', 'best');
title('Nonsmooth Curve Fitting');