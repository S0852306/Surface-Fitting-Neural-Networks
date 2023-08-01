# Function Approximation Neural Networks
**Neural networks for multivariable function approximation & classification.**
# Introduction
* Neural networks are universal function approximators, which means that given enough parameters, a neural net can approximate any multivariable continuous function to any desired level of accuracy.
* Many built-in functions in MATLAB only provide functionality for curve fitting (1D) or surface fitting (2D). In contrast, neural networks can handle more general and higher-dimensional fitting problems. 
* More Info: https://tinyurl.com/wre9r5uk 
(MATLAB File Exchange)

# User Guide
"GeneralGuide.mlx" provide a general workflow and detailed instructions on how to configure the solver.
If you are not familiar with numerical optimization or deep learning, you can also use the following command to automatically configure the solver.

[![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=S0852306/Implementing-neural-networks-from-scratch&file=&file=<GeneralGuide.mlx>)
```
% Network Structure Set Up, for regression
LayerStruct=[InputDimension,10,10,10,OutputDimension];
NN=Initialization(LayerStruct);
% Solver Set Up
option.MaxIteration=500;
NN=OptimizationSolver(data,label,NN,option);
```
"DigitRecognition.mlx" utilizes a simple MLP architecture and achieves an accuracy of **97.6%** on the testing set of the "MNIST" handwritten digit recognition dataset.
```
% Network Structure Set Up, for classification
NN.Cost='Entropy';
LayerStruct=[InputDimension,10,10,10,OutputDimension];
NN=Initialization(LayerStruct,NN);
% Solver Set Up
option.Solver='ADAM'; option.s0=1e-3; % step size option.BatchSize=512;
option.MaxIteration=30; % number of epoch
NN=OptimizationSolver(data,label,NN,option);
```
![LogoFitR](https://github.com/S0852306/Implementing-Neural-Networks-from-Scratch./assets/111946393/5c7e86e9-cfde-44e6-a69c-af08dfafafa5)

![MNIST7](https://github.com/S0852306/Implementing-Neural-Networks-from-Scratch./assets/111946393/a9c92ab7-e3a2-4948-8f16-c73b106781d2)

# Types of Neural Nets
 1. Ordinary MultiLayer Perceptron 
 2. Residual Neural Network
# Optimization Solvers
 1. Stochastic Gradient Descents (SGD)
 2. Stochastic Gradient Descents with Momentum (SGDM)
 3. Adaptive Momentum Estimation (ADAM)
 4. Root Mean Square Propagation (RMSprop)
 5. Broyden-Fletcher-Goldfarb-Shanno Method (BFGS)

# Reference
 1. Numerical Optimization, Nocedal & Wright.
 2. Practical Quasi-Newton Methods for Training Deep Neural Networks, Goldfarb, et al.
 3. Kronecker-factored Quasi-Newton Methods for Deep Learning, Yi Ren, et al.
