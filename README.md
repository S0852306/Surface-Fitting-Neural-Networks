# Function Approximation Neural Networks
**Neural networks for multivariable function approximation & classification.**
# Introduction
* Neural networks are universal function approximators, which means that given enough parameters, a neural net can approximate any multivariable continuous function to any desired level of accuracy.
* The hybrid optimization algorithms used in the pack are specially designed for scientific computing tasks, outperforming several state-of-the-art first-order methods such as ADAM.
* More Details in MATLAB File Exchange : [Surface Fitting using Neural Networks](https://tinyurl.com/wre9r5uk)  

# User Guide
"GeneralGuide.mlx" provides a general workflow and detailed instructions on configuring the solver.
If you are unfamiliar with numerical optimization or deep learning, you can use the following command to configure the solver automatically.

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
LayerStruct=[InputDimension,128,64,16,OutputDimension];
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
 4. Adaptive Momentum Estimation with Weight Decay (AdamW)
 5. Root Mean Square Propagation (RMSprop)
 6. Broyden-Fletcher-Goldfarb-Shanno Method (BFGS)

# Reference
 1. Numerical Optimization, Nocedal & Wright.
 2. Practical Quasi-Newton Methods for Training Deep Neural Networks, Goldfarb, et al.
 3. Kronecker-factored Quasi-Newton Methods for Deep Learning, Yi Ren, et al.
