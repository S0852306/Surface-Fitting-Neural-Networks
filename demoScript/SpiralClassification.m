clear; clc; close all;

%% ------------------------------------------------------------
% 1) Generate spiral-like dataset via linear ODE
%% ------------------------------------------------------------
A = [-0.1 -1; 1 -0.1];
odeFun = @(t, x) A*x;

numPerClass = 400;
T = 20;
t = linspace(0, T, numPerClass);

% class 1
x0 = [2; 0];
[~, x] = ode45(odeFun, t, x0);   % (numPerClass x 2)

% class 2
y0 = [1.5; 0];
[~, y] = ode45(odeFun, t, y0);   % (numPerClass x 2)

%% ------------------------------------------------------------
% 2) Add noise
%% ------------------------------------------------------------
noiseStd = 0.01;
x = x + noiseStd * randn(size(x));
y = y + noiseStd * randn(size(y));

%% ------------------------------------------------------------
% 3) Build training data / one-hot labels (regression targets)
%% ------------------------------------------------------------
data = [x; y].';                 % (2 x 2*numPerClass)

label = zeros(2, 2*numPerClass);
label(1, 1:numPerClass) = 1;          % class 1 -> [1;0]
label(2, numPerClass+1:end) = 1;      % class 2 -> [0;1]

%% ------------------------------------------------------------
% 4) Train NN (MSE regression)
%% ------------------------------------------------------------
LayerStruct = [2, 20, 20, 20, 2];

NN = struct();
NN.Cost = 'Entropy';
NN.ActivationFunction = 'ReLU';
NN.NetworkType = 'ResNet';
NN = Initialization(LayerStruct, NN);

option = struct();
option.Solver = 'ADAM';
option.s0 = 2e-3;
option.MaxIteration = 800;

NN = OptimizationSolver(data, label, NN, option);

%% ------------------------------------------------------------
% 5) Figure 1: Decision region + boundary
%% ------------------------------------------------------------
figure; hold on; grid on;

plotDecisionBoundary2D(NN, x, y);

h1 = scatter(x(:,1), x(:,2), 18, 'filled');  % Class 1
h2 = scatter(y(:,1), y(:,2), 18, 'filled');  % Class 2

legend([h1 h2], {'Class 1','Class 2'}, 'Location','best');
xlabel('x1'); ylabel('x2');
title('Decision Boundary (argmax of 2 outputs)');

%% ------------------------------------------------------------
% 6) Figure 2: 2D SCORE MAP (imagesc)
%% ------------------------------------------------------------
figure;

[XX, YY, out1Map, out2Map] = evaluateGrid2D(NN, x, y, 300, 0.5);

xList = XX(1,:);
yList = YY(:,1);

% -------------------------
% Output 1 heatmap
% -------------------------
subplot(1,2,1);
imagesc(xList, yList, out1Map);
axis xy; axis tight; colorbar;
hold on;
scatter(x(:,1), x(:,2), 10, 'k', 'filled');
scatter(y(:,1), y(:,2), 10, 'k', 'filled');
xlabel('x1'); ylabel('x2');
title('Output-1 Score Map');

% -------------------------
% Output 2 heatmap
% -------------------------
subplot(1,2,2);
imagesc(xList, yList, out2Map);
axis xy; axis tight; colorbar;
hold on;
scatter(x(:,1), x(:,2), 10, 'k', 'filled');
scatter(y(:,1), y(:,2), 10, 'k', 'filled');
xlabel('x1'); ylabel('x2');
title('Output-2 Score Map');

%% ============================================================
% Local functions
%% ============================================================
function plotDecisionBoundary2D(NN, x, y)
    [XX, YY, out1Map, out2Map] = evaluateGrid2D(NN, x, y, 300, 0.5);

    % classify by argmax(out1,out2)
    predMap = ones(size(out1Map));
    predMap(out2Map > out1Map) = 2;

    xList = XX(1,:);
    yList = YY(:,1);

    % background
    hImg = imagesc(xList, yList, predMap);
    set(hImg, 'AlphaData', 0.15);
    axis xy;

    % boundary (where class changes)
    hC = contour(XX, YY, double(predMap), [1.5 1.5], 'k', 'LineWidth', 2);
    try
        hC.HandleVisibility = 'off';
    catch
    end

    try
        uistack(hImg, 'bottom');
    catch
    end
end

function [XX, YY, out1Map, out2Map] = evaluateGrid2D(NN, x, y, gridN, pad)
    if nargin < 4 || isempty(gridN), gridN = 300; end
    if nargin < 5 || isempty(pad),   pad = 0.5;   end

    allPts = [x; y];
    xMin = min(allPts(:,1)); xMax = max(allPts(:,1));
    yMin = min(allPts(:,2)); yMax = max(allPts(:,2));

    xList = linspace(xMin-pad, xMax+pad, gridN);
    yList = linspace(yMin-pad, yMax+pad, gridN);
    [XX, YY] = meshgrid(xList, yList);

    gridData = [XX(:)'; YY(:)'];      % (2 x M)
    out = NN.Evaluate(gridData);      % (2 x M)

    out1Map = reshape(out(1,:), size(XX));
    out2Map = reshape(out(2,:), size(XX));
end