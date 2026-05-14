function OptimizedNN=OptimizationSolver(data,label,NN,option)
% v1.1.8

NN.OptimizationHistory=zeros(2,1);
NN.StepSizeHistory=zeros(2,1);
NN.LineSearchIteration=zeros(2,1);
NN.numOfData=size(data,2); NN.MeanFactor=1/size(data,2);

if strcmp(NN.Cost,'Entropy')==1
    NN.MeanFactor=1/size(data,2);
elseif strcmp(NN.Cost,'MSE')==1
    NN.MeanFactor=2/size(data,2);
elseif strcmp(NN.Cost,'MAE')==1
    NN.MeanFactor=1/size(data,2);    
elseif strcmp(NN.Cost,'SSE')==1
    NN.MeanFactor=2;
end

if isfield(option,'Solver')==0 && strcmp(NN.Cost,'Entropy')==0
    option.Solver='Auto';
elseif isfield(option,'Solver')==0
    option.Solver='ADAM';
end

if isfield(option,'Solver')==0
    option.Solver='Auto';
end
solver=option.Solver;
NN.Solver=option.Solver;

if isfield(option,'s0')==0
    option.s0=2e-3;
end

if isfield(option,'BatchSize')==0
    option.BatchSize=round(size(data,2)/10);
end

if strcmp(NN.InputAutoScaling,'on')
    InputScaleVector=std(data,0,2);
    InputCenterVector=mean(data,2);
    NN.InputCenterVector= InputCenterVector./InputScaleVector;
    NN.InputScaleVector= 1./InputScaleVector;
else
    NN.InputCenterVector = 0;
    NN.InputScaleVector = 1;
end

if strcmp(NN.LabelAutoScaling,'on')
    LabelScaleVector=std(label,0,2);
    LabelCenterVector=mean(label,2);
    label=(label-LabelCenterVector)./LabelScaleVector;
    NN.LabelCenterVector=LabelCenterVector;
    NN.LabelScaleVector=LabelScaleVector;
else
    NN.LabelCenterVector = 0;
    NN.LabelScaleVector = 1;
end

if isfield(NN,'activeDerivate')==0
    disp('Please provide the derivatives of activation functions.');
end

WeightedFlag=isfield(option,'weighted');
if WeightedFlag==1
    NN.SampleWeight=[];
    NN.Weighted=option.weighted;
    NN.WeightedFlag=1;
else
    NN.WeightedFlag=0;
end

% ===================== 1) Add 'LBFGS' to the top-level solver switch =====================
switch solver
    case 'BFGS'
        OptimizedNN = QuasiNewtonSolver(data, label, NN, option);
    case 'LBFGS'
        OptimizedNN = QuasiNewtonSolver(data, label, NN, option);
    case 'AdamW'
        OptimizedNN = StochasticSolver(data, label, NN, option);
    case 'ADAM'
        OptimizedNN = StochasticSolver(data, label, NN, option);
    case 'SGDM'
        OptimizedNN = StochasticSolver(data, label, NN, option);
    case 'SGD'
        OptimizedNN = StochasticSolver(data, label, NN, option);
    case 'RMSprop'
        OptimizedNN = StochasticSolver(data, label, NN, option);
    case 'Auto'
        %------------ First Stage Optimization ----------------------
        tic
        if isfield(option,'MaxIteration')==1
            TotalIteration=option.MaxIteration;
        else
            TotalIteration=800;
        end
        option.Solver='ADAM';
        option.s0=2e-3;
        option.MaxIteration=round(TotalIteration/4);
        option.BatchSize=round(size(data,2)/10);
        NN=StochasticSolver(data,label,NN,option);

        disp('------------------------------------------------------')
        DisplayWord=['First Stage Optimization Finished in  ', num2str(option.MaxIteration), '  Iteration.'];
        disp(DisplayWord)
        disp('------------------------------------------------------')

        %------------ Second Stage Optimization ----------------------
        option.Solver='BFGS';
        option.MaxIteration=TotalIteration-round(TotalIteration/4);
        OptimizedNN=QuasiNewtonSolver(data,label,NN,option);
        NN.OptimizationTime=toc;
end

NetworkType=NN.NetworkType;
switch NetworkType
    case'ANN'
        Net=@(x,NN) ANN(x,NN);
    case 'ResNet'
        Net=@(x,NN) ResNet(x,NN);
    case'SirenNet'
        Net=@(x,NN) SirenNet(x,NN);
end

if strcmp(NN.LabelAutoScaling,'on')==1
    OptimizedNN.Evaluate=@(x) NN.LabelScaleVector.*Net(x,OptimizedNN)+NN.LabelCenterVector;
    Error=(NN.LabelScaleVector.*label+NN.LabelCenterVector)-OptimizedNN.Evaluate(data);
else
    OptimizedNN.Evaluate=@(x) Net(x,OptimizedNN);
    Error=label-OptimizedNN.Evaluate(data);
end

if ~strcmp(NN.Cost,'Entropy')
    OptimizedNN.Derivate = @(x) AutomaticDerivate(x,OptimizedNN);
    OptimizedNN.MeanAbsoluteError = sum(abs(Error),[1 2]) / NN.numOfData;

    disp('------------------------------------------------------')
    FormatSpec = 'Max Iteration : %d , Cost : %16.8f \n';
    FinalCost = CostFunction(data,label,OptimizedNN);
    fprintf(FormatSpec,OptimizedNN.Iteration,FinalCost);
    fprintf('Optimization Time : %5.1f\n',OptimizedNN.OptimizationTime);
    fprintf('Mean Absolute Error : %8.4f\n',OptimizedNN.MeanAbsoluteError)
    disp('------------------------------------------------------')
else
    OptimizedNN.ComputeAccuracy = @(d,l) ComputeAccuracy(d,l,OptimizedNN);
    OptimizedNN.Predict = @(d) ClassPredict(d,OptimizedNN);
    Accuracy = ComputeAccuracy(data,label,OptimizedNN);
    OptimizedNN.Accuracy = Accuracy;

    disp('------------------------------------------------------')
    FormatSpec = 'Max Iteration : %d , Cost : %16.8f \n';
    FinalCost = CostFunction(data,label,OptimizedNN);
    fprintf(FormatSpec,OptimizedNN.Iteration,FinalCost);
    fprintf('Accuracy : %6.2f %% \n',OptimizedNN.Accuracy);
    fprintf('Optimization Time : %5.1f\n',OptimizedNN.OptimizationTime);
    disp('------------------------------------------------------')
end