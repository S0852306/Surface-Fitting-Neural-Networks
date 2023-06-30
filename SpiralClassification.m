clear; clc; close all;
%%  Generate Spiral Data Set
% Class 1
x0 = [2; 0];
A = [-0.1 -1; 1 -0.1];
ode=@(t,x) A*x;
NumOfSample=400;
T=10;
t=linspace(0,T,NumOfSample);
[~, x] = ode45(ode, t, x0);
%%
% Class 2
y0 = [1.5; -0];
NumOfSample=400;
[~, y] = ode45(ode, t, y0);
%%  Add Random Noise
mag=0.05;
x=x+mag*randn(size(x));
y=y+mag*randn(size(y));
%% Visualization
scatter(x(:,1),x(:,2))
hold on
scatter(y(:,1),y(:,2))
legend('Class 1','Class 2')
%% One Hot Encoding
data=[x;y]';
label=zeros(2,2*NumOfSample);
label(1,1:NumOfSample)=ones(1,NumOfSample);
label(2,NumOfSample+1:end)=ones(1,NumOfSample);
%%
LayerStruct=[2,20,20,20,2];
NN.Cost='Entropy';
NN.ActivationFunction='ReLU';
NN=Initialization(LayerStruct,NN);
%%
option.Solver='ADAM';
option.s0=2e-3;
option.MaxIteration=800;
NN=OptimizationSolver(data,label,NN,option);
%%

