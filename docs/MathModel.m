%% Mathematical Model of Neural Networks
% Demonstrates the forward pass of a neural network and verifies it
% step by step. Also covers auto-scaling (normalization).
%
%   h0 = x
%   hk = sigma(Wk * h_{k-1} + bk),  k = 1,...,d-1
%   y  = Wd * h_{d-1} + bd
%
% The default activation is Gaussian: sigma(z) = exp(-z^2).
% B-spline activation replaces it with a learnable piecewise polynomial.

clear; clc; close all;

%% Generate Data
% y = x1^2 + x2^2

n = 20;
xspace = linspace(-2, 2, n);
yspace = linspace(-2, 2, n);
count = 0;
for j = 1:n
    for k = 1:n
        count = count + 1;
        Input(1, count) = xspace(j);
        Input(2, count) = yspace(k);
        zdata(count) = xspace(j)^2 + yspace(k)^2;
    end
end

scatter3(Input(1,:), Input(2,:), zdata, 6, zdata, 'filled');
title('y = x_1^2 + x_2^2');
xlabel('x_1'); ylabel('x_2'); zlabel('y = f(x_1, x_2)');

%% Train Network
% Architecture: [2, 3, 4, 1], Gaussian activation.

InSize = 2; OutSize = 1;
LayerStruct = [InSize, 3, 4, OutSize];
NN = Initialization(LayerStruct);
option.MaxIteration = 500;
NN = OptimizationSolver(Input, zdata, NN, option);

%% Network Architecture
% length(LayerStruct)-1 = depth; each entry = layer width.

depth = length(LayerStruct) - 1;
fprintf('Depth d = %d\n', depth);
fprintf('W1: %dx%d,  W2: %dx%d,  W3: %dx%d\n', ...
    size(NN.weight{1},1), size(NN.weight{1},2), ...
    size(NN.weight{2},1), size(NN.weight{2},2), ...
    size(NN.weight{3},1), size(NN.weight{3},2));

%% Accessing Parameters

W1 = NN.weight{1}  
W2 = NN.weight{2}  
b1 = NN.bias{1}

%% Point-wise Activation
% sigma is applied element-wise: sigma([a;b;c]) = [sigma(a);sigma(b);sigma(c)]

DemoActive = NN.active([pi; 0; 2.5]);
PointWiseMATLAB = exp(-[pi; 0; 2.5].^2);
disp('  NN.active     exp(-z^2)');
disp([DemoActive, PointWiseMATLAB]);

%% Step-by-Step Forward Pass
% Goal: NN([0.2; 0.5]) ~ 0.2^2 + 0.5^2 = 0.29

h0 = [0.2; 0.5];
h1 = NN.active(NN.weight{1}*h0 + NN.bias{1}) 

% Layer 2

h2 = NN.active(NN.weight{2}*h1 + NN.bias{2})

% Output layer (linear, no activation)

z = NN.weight{3}*h2 + NN.bias{3}
zTrue = 0.2^2 + 0.5^2;
fprintf('Manual forward:  %.4f\nNN.Evaluate:     %.4f\nTrue value:      %.4f\n', ...
    z, NN.Evaluate([0.2; 0.5]), zTrue);

%% Preprocessing (Normalization)
% x' = (x - mean(x)) / std(x)

xDemo = linspace(0, 2*pi, 10);
xNormalized = (xDemo - mean(xDemo)) / std(xDemo);
fprintf('Before: [%.2f, ..., %.2f]\n', xDemo(1), xDemo(end));
fprintf('After:  [%.4f, ..., %.4f]\n', xNormalized(1), xNormalized(end));

%% Train with Auto-Scaling
% NN.Evaluate compensates normalization:
%   h0 = s1.*x - c1,  y = s2.*hd + c2

NN.InputAutoScaling = 'on';
NN.LabelAutoScaling = 'on';
NN = Initialization(LayerStruct, NN);
NN = OptimizationSolver(Input, zdata, NN, option);

%% Scaling Parameters

s1 = NN.InputScaleVector   
c1 = NN.InputCenterVector  
s2 = NN.LabelScaleVector   
c2 = NN.LabelCenterVector  

%% Forward Pass with Auto-Scaling
% Same goal: NN([0.2; 0.5]) ~ 0.29

h0 = s1.*[0.2; 0.5] - c1;                       % scaled input
h1 = NN.active(NN.weight{1}*h0 + NN.bias{1});   % layer 1
h2 = NN.active(NN.weight{2}*h1 + NN.bias{2});   % layer 2
h3 = NN.weight{3}*h2 + NN.bias{3};              % output (linear)
zScaled = s2.*h3 + c2;                           % rescale

fprintf('Manual (scaled): %.4f\nNN.Evaluate:     %.4f\nTrue value:      %.4f\n', ...
    zScaled, NN.Evaluate([0.2; 0.5]), zTrue);
