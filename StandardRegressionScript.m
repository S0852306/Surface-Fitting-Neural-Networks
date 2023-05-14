clear; clc; close all;
%% Import Data
n=40;
x=linspace(-2,2,n);
y=linspace(-2,2,n);
n1=numel(x); n2=numel(y);
count=0;
    for j=1:n1
        for k=1:n2
            count=count+1;
            data(1,count)=x(j);
            data(2,count)=y(k);

            u=0.5*x(j)^2+0.7*y(k)^2; 
            label(1,count)=exp(-u/4)*cos(2*u);
            label(2,count)=0.5+(sin(x(j)^2+y(k)*x(j))^2-0.5)/((1+0.001*y(k))^2);
            % x1=x(j); u=y(k);
            % label(3,count)=-2*x1*cos(x1)/sqrt(1+u^2)+(sin(u)-2*u)*abs(x1)/10;

        end
    end
% [data(1,:),avg1,std1]=normalize(data(1,:));
% [data(2,:),avg2,std2]=normalize(data(2,:));

% n_size = 15;
% n=n_size;
% x = 1:n_size*2+1;
% y = 1:n_size*2+1;
% [X,Y] = meshgrid(x,y);
% V = membrane(1,n_size);
% predictor = [X(:),Y(:)]';
% response = V(:)';
% 
% data=predictor;
% label=response;

%% Network Structure Set Up
InputDimension=2; OutputDimension=2;
LayerStruct=[InputDimension,5,5,8,OutputDimension];
NN.ActivationFunction='Gaussian';
NN.Cost='SSE';
NN.NetworkType='ANN';
NN=Initialization(NN,LayerStruct);

%% First Order Solver Set Up

option.Solver='ADAM';
option.s0=15e-3;
option.MaxIteration=200;
option.BatchSize=200;
NN=OptimizationSolver(data,label,NN,option);
%% Quasi-Newton Solver Set Up
option.Solver='BFGS';
option.MaxIteration=300;
option.TerminateCondition=1e-8;
NN=OptimizationSolver(data,label,NN,option);
%% Validation
Prediction=NN.Evaluate(data);
Error=label-Prediction;

%% Statistics
figure; histogram(Error);
title('Error Distribution')
figure
scatter3(data(1,:),data(2,:),label(1,:),'black')
hold on
[X,Y]=meshgrid(x,y);

surf(X,Y,reshape(Prediction(1,:),n1,n2))
title('Neural Network Fit')
legend('Data','Fitting')