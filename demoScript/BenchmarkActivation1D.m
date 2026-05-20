clear; clc; close all;

x = linspace(-3, 3, 160);
y = sign(sin(3*x)) + 0.15*cos(11*x);

bsplineModel = NeuralFit(x, y, [1, 1], 'basis', 'BSpline');
gaussianModel = NeuralFit(x, y, [1, 1]);

yBspline = bsplineModel.Evaluate(x);
yGaussian = gaussianModel.Evaluate(x);

errBspline = abs(yBspline - y);
errGaussian = abs(yGaussian - y);

figure();
subplot(2,1,1);
plot(x, y, 'k.', 'MarkerSize', 8, 'DisplayName', 'Data'); hold on;
plot(x, yBspline, 'r-', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('B-spline (MAE=%.2e)', mean(errBspline)));
plot(x, yGaussian, 'b--', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('Gaussian (MAE=%.2e)', mean(errGaussian)));
grid on;
legend('Location', 'best');
title('Fitting Comparison');
ylabel('y');

subplot(2,1,2);
semilogy(x, errBspline, 'r-', 'LineWidth', 1.2, ...
    'DisplayName', 'B-spline |error|'); hold on;
semilogy(x, errGaussian, 'b--', 'LineWidth', 1.2, ...
    'DisplayName', 'Gaussian |error|');
grid on;
legend('Location', 'best');
xlabel('x'); ylabel('|error|');
title('Pointwise Absolute Error');
