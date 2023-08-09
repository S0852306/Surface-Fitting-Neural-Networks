clear; clc; close all;
%% Import data
n=20;
x=linspace(-2,2,n);
y=linspace(-2,2,n);
n1=numel(x); n2=n1;
count=0;
for j=1:n2
    for k=1:n1
        count=count+1;
        data(1,count)=x(j);
        data(2,count)=y(k);
        u=x(j)^2+y(k)^2;
        label(1,count)=(log(1+(x(k)-4/3)^2+3*(x(k)+y(j)-x(k)^3)^2));
        label(2,count)=exp(-u/2).*cos(2*u);
    end
end



%% Network Structure Set Up
InputDimension=2; OutputDimension=2;
LayerStruct=[InputDimension,10,10,10,OutputDimension];
NN.ActivationFunction='Gaussian';
NN.Cost='SSE';
NN.NetworkType='ANN';
NN.InputAutoScaling='off';
NN.labelAutoScaling='on';
NN=Initialization(LayerStruct,NN);

%% First Order Solver Set Up


option.Solver='ADAM';
option.s0=1e-3;
option.MaxIteration=250;
option.BatchSize=100;
NN=OptimizationSolver(data,label,NN,option);

%% Quasi-Newton Solver Set Up

option.Solver='BFGS';
option.MaxIteration=500;
NN=OptimizationSolver(data,label,NN,option);



%%
plot(log10(NN.OptimizationHistory))

%% Validation
Prediction=NN.Evaluate(data);
Error=label-Prediction;
Report=FittingReport(data,label,NN);
%% Statistics
figure
slice=2;
scatter3(data(1,:),data(2,:),label(slice,:),'black')
hold on
[X,Y]=meshgrid(x,y);
n1=numel(x); n2=numel(y);
surf(X,Y,reshape(Prediction(slice,:),n1,n2))
title('Neural Network Fit')
legend('Data','Fitting')
