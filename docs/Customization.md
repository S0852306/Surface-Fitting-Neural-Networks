# Customization Guide

NeuralNetsPack provides two training workflows:

## Simplified API — `NeuralFit`

One-line training with sensible defaults. Suitable for most fitting tasks:

```matlab
NN = NeuralFit(data, label, [inputDim, outputDim]);
prediction = NN.Evaluate(newData);
```

Configurable via name-value pairs:

```matlab
NN = NeuralFit(data, label, [N, M], ...
    'basis', 'Gaussian', ...
    'stage1Iterations', 100, ...
    'stage2Iterations', 500);
```

Internally uses a `[N, 10, 10, 10, 10, M]` architecture, ADAM → BFGS two-stage optimizer, and auto-scaling.

(`'activation'` is accepted as an alias for `'basis'`.)

## Full Customizable Training — `Initialization` + `OptimizationSolver`

For control over architecture, network type, solver, and all hyperparameters:

```matlab
LayerStruct = [2, 30, 30, 30, 1];   % custom depth and width
NN.NetworkType = 'ResNet';           % or 'ANN'
NN.InputAutoScaling = 'on';
NN.LabelAutoScaling = 'on';
NN = Initialization(LayerStruct, NN);

option.Solver = 'ADAM';
option.MaxIteration = 300;
option.s0 = 1e-3;
NN = OptimizationSolver(data, label, NN, option);

option.Solver = 'LBFGS';
option.MaxIteration = 500;
NN = OptimizationSolver(data, label, NN, option);
```

The following sections detail each configurable axis.

---

## 1. Activation Function

The activation determines the nonlinearity applied at each hidden layer.

### Via NeuralFit

```matlab
% Default: learnable B-spline
NN = NeuralFit(data, label, [N, M]);

% Fixed Gaussian basis
NN = NeuralFit(data, label, [N, M], 'basis', 'Gaussian');

% Other fixed bases
NN = NeuralFit(data, label, [N, M], 'basis', 'tanh');
NN = NeuralFit(data, label, [N, M], 'basis', 'ReLU');
NN = NeuralFit(data, label, [N, M], 'basis', 'Sigmoid');
NN = NeuralFit(data, label, [N, M], 'basis', 'Wavelet');
NN = NeuralFit(data, label, [N, M], 'basis', 'Sine');
```

### Via Initialization (full control)

```matlab
LayerStruct = [2, 20, 20, 1];
NN = Initialization(LayerStruct);            % Gaussian by default
NN.activationID = 3;                         % switch to tanh
NN = Initialization(LayerStruct, NN);
```

### Available Activations

| Name | ID | Formula |
|------|----|---------|
| Gaussian | 1 | exp(-z^2) |
| Sigmoid | 2 | 1/(1+exp(-z)) |
| tanh | 3 | tanh(z) |
| ReLU | 4 | max(0, z) |
| Wavelet | 5 | (1-z^2) exp(-z^2/2) |
| Sine | 6 | sin(z) |
| BSpline | 7 | learnable piecewise polynomial |

### B-Spline Configuration

When using BSpline, the activation shape is learned during training. Configuration:

```matlab
NN.bspline.order   = 3;         % 3 = quadratic, 4 = cubic
NN.bspline.numGrid = 8;         % number of grid intervals
NN.bspline.gridRange = [-4, 4]; % domain of knot vector
NN.bspline.initShape = 'Gaussian'; % initial shape approximation
```

Valid `initShape` values: `'Gaussian'`, `'Sigmoid'`, `'tanh'`, `'ReLU'`, `'Identity'`, `'SiLU'`, `'Wavelet'`.

### Custom Activation (function handle)

```matlab
NN.active = @(z) sin(z)./z;
NN.activeDerivate = @(z, a) (cos(z) - a)./z;
NN.activationID = 0;  % signals custom
```

---

## 2. Layer Structure

The network depth and width are defined by a single vector.

```matlab
% [inputDim, hidden1, hidden2, ..., outputDim]
LayerStruct = [3, 50, 50, 50, 2];  % 3 hidden layers, width 50
NN = Initialization(LayerStruct);
```

- **Depth** = `length(LayerStruct) - 2` hidden layers
- **Width** = value of each interior element
- The first element must match the input dimension
- The last element must match the output dimension

`NeuralFit` uses a fixed `[N, 10, 10, 10, 10, M]` structure internally. For different architectures, use the direct workflow:

```matlab
LayerStruct = [1, 30, 30, 1];
NN = Initialization(LayerStruct);
option.MaxIteration = 800;
NN = OptimizationSolver(data, label, NN, option);
```

---

## 3. Network Type

### Standard Feedforward (ANN)

```matlab
NN.NetworkType = 'ANN';  % default
```

Each layer: $\mathbf{h}_k = \sigma(\mathbf{W}_k \mathbf{h}_{k-1} + \mathbf{b}_k)$.

### Residual Network (ResNet)

```matlab
NN.NetworkType = 'ResNet';
NN = Initialization(LayerStruct, NN);
```

Adds identity skip connections where dimensions match. Useful for deeper networks (5+ hidden layers) to avoid vanishing gradients.

---

## 4. Optimizer Configuration

### Two-Stage Training (NeuralFit default)

```matlab
NN = NeuralFit(data, label, [N, M], ...
    'stage1Iterations', 50, ...   % ADAM (stochastic)
    'stage2Iterations', 550);     % BFGS (quasi-Newton)
```

Stage 1 (ADAM) provides a rough global search. Stage 2 (BFGS/LBFGS) refines to high accuracy.

### Direct Solver Control

```matlab
option.Solver = 'ADAM';       % or 'BFGS', 'LBFGS', 'AdamW', 'SGDM', 'RMSprop'
option.MaxIteration = 500;
option.s0 = 1e-3;            % learning rate
option.BatchSize = 64;
NN = OptimizationSolver(data, label, NN, option);
```

### Key Options

| Field | Default | Description |
|-------|---------|-------------|
| `Solver` | `'Auto'` | Solver selection |
| `MaxIteration` | 500 | Iteration limit |
| `s0` | 2e-3 | Step size / learning rate |
| `BatchSize` | N/10 | Mini-batch size (stochastic only) |
| `lbfgsMemory` | 10 | L-BFGS history pairs |
| `TerminateCondition` | 1e-5 | Convergence tolerance (QN only) |

---

## 5. Preprocessing

```matlab
NN.InputAutoScaling = 'on';   % normalize inputs
NN.LabelAutoScaling = 'on';   % normalize outputs
```

Auto-scaling applies z-score normalization before training and compensates automatically in `NN.Evaluate`. Enabled by default in `NeuralFit`.

---

## 6. Cost Function

```matlab
NN.Cost = 'MSE';     % mean squared error (default for regression)
NN.Cost = 'Entropy'; % cross-entropy (classification)
NN.Cost = 'MAE';     % mean absolute error
```

---

## Complete Example: Franke's Function with Deep ResNet + Cubic B-Spline

Franke's function is a standard 2-D test surface with peaks, valleys, and saddle regions:

```matlab
% Generate Franke's function on scattered points
rng(0);
xy = rand(2, 2000);  % 2000 random points in [0,1]^2
x1 = xy(1,:); x2 = xy(2,:);
label = 0.75*exp(-((9*x1-2).^2 + (9*x2-2).^2)/4) ...
      + 0.75*exp(-(9*x1+1).^2/49 - (9*x2+1)/10) ...
      + 0.50*exp(-((9*x1-7).^2 + (9*x2-3).^2)/4) ...
      - 0.20*exp(-(9*x1-4).^2 - (9*x2-7).^2);

LayerStruct = [2, 30, 30, 30, 30, 1];
NN.NetworkType = 'ResNet';
NN.InputAutoScaling = 'on';
NN.LabelAutoScaling = 'on';
NN.bspline.order = 4;         % cubic
NN.bspline.numGrid = 12;
NN.bspline.initShape = 'Gaussian';
NN = Initialization(LayerStruct, NN);

% Stage 1: ADAM
opt1.Solver = 'ADAM';
opt1.MaxIteration = 200;
opt1.s0 = 1e-3;
NN = OptimizationSolver(xy, label, NN, opt1);

% Stage 2: L-BFGS
opt2.Solver = 'LBFGS';
opt2.MaxIteration = 500;
NN = OptimizationSolver(xy, label, NN, opt2);

prediction = NN.Evaluate(xy);
fprintf('MAE: %.6f\n', mean(abs(prediction - label)));
```
```
