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
    NN.InputCenterVector=InputCenterVector./InputScaleVector;
    NN.InputScaleVector=1./InputScaleVector;
end

if strcmp(NN.LabelAutoScaling,'on')
    LabelScaleVector=std(label,0,2);
    LabelCenterVector=mean(label,2);
    label=(label-LabelCenterVector)./LabelScaleVector;
    NN.LabelCenterVector=LabelCenterVector;
    NN.LabelScaleVector=LabelScaleVector;
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

switch solver
    case 'BFGS'
        OptimizedNN=QuasiNewtonSolver(data,label,NN,option);
    case 'AdamW'
        OptimizedNN=StochasticSolver(data,label,NN,option);
    case 'ADAM'
        OptimizedNN=StochasticSolver(data,label,NN,option);
    case 'SGDM'
        OptimizedNN=StochasticSolver(data,label,NN,option);
    case 'SGD'
        OptimizedNN=StochasticSolver(data,label,NN,option);
    case 'RMSprop'
        OptimizedNN=StochasticSolver(data,label,NN,option);
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
end

if strcmp(NN.LabelAutoScaling,'on')==1
    OptimizedNN.Evaluate=@(x) NN.LabelScaleVector.*Net(x,OptimizedNN)+NN.LabelCenterVector;
    Error=(NN.LabelScaleVector.*label+NN.LabelCenterVector)-OptimizedNN.Evaluate(data);
else
    OptimizedNN.Evaluate=@(x) Net(x,OptimizedNN);
    Error=label-OptimizedNN.Evaluate(data);
end

if strcmp(NN.Cost,'Entropy')==0
    OptimizedNN.Derivate=@(x) AutomaticDerivate(x,OptimizedNN);
    OptimizedNN.MeanAbsoluteError=sum(abs(Error),[1 2])/NN.numOfData;
    
    disp('------------------------------------------------------')
    FormatSpec = 'Max Iteration : %d , Cost : %16.8f \n';
    FinalCost=CostFunction(data,label,OptimizedNN);
    fprintf(FormatSpec,OptimizedNN.Iteration,FinalCost);
    fprintf('Optimization Time : %5.1f\n',OptimizedNN.OptimizationTime);
    fprintf('Mean Absolute Error : %8.4f\n',OptimizedNN.MeanAbsoluteError)
    disp('------------------------------------------------------')
    
else

    OptimizedNN.ComputeAccuracy=@(data,label) ComputeAccuracy(data,label,OptimizedNN);
    Accuracy=OptimizedNN.ComputeAccuracy(data,label);
    OptimizedNN.Predict=@(data) ClassPredict(data,OptimizedNN);
    OptimizedNN.Accuracy=Accuracy;
    
    disp('------------------------------------------------------')
    FormatSpec = 'Max Iteration : %d , Cost : %16.8f \n';
    FinalCost=CostFunction(data,label,OptimizedNN);
    fprintf(FormatSpec,OptimizedNN.Iteration,FinalCost);
    fprintf('Accuracy : %6.2f %% \n',OptimizedNN.Accuracy);
    fprintf('Optimization Time : %5.1f\n',OptimizedNN.OptimizationTime);
    disp('------------------------------------------------------')
end
%% Numerical Optimization Solver

    function OptimizedNN=StochasticSolver(data,label,NN,option)
        Counter=0;

        % ------------- Select Gradient Solver----------------
        NetworkType=NN.NetworkType;
        if isfield(option,'GradientSolver')==0
            switch NetworkType
                case 'ANN'
                    AutoGrad=@(data,label,NN) AutomaticGradient(data,label,NN);
                case'ResNet'
                    AutoGrad=@(data,label,NN) ElementWiseRNAG(data,label,NN);
            end
        else
            switch NetworkType
                case 'ANN'
                    GradientSolver=option.GradientSolver;
                    switch GradientSolver
                        case 'Element'
                            AutoGrad=@(data,label,NN) ElementWiseAG(data,label,NN);
                        case 'Column'
                            AutoGrad=@(data,label,NN) ColumnWiseAG(data,label,NN);
                        case 'General'
                            AutoGrad=@(data,label,NN) ComplexStepGradient(data,label,NN);
                    end
                case'ResNet'
                    GradientSolver=option.GradientSolver;
                    switch GradientSolver
                        case 'Element'
                            AutoGrad=@(data,label,NN) ElementWiseRNAG(data,label,NN);
                        case 'Column'
                            AutoGrad=@(data,label,NN) ColumnWiseRNAG(data,label,NN);
                        case 'General'
                            AutoGrad=@(data,label,NN) ComplexStepGradient(data,label,NN);
                    end
            end
        end
        % ------------- Select Gradient Solver----------------
        BatchCost=zeros(2,1);
        BatchSize=option.BatchSize;
        tic
        for j=1:option.MaxIteration
            NN.Iteration=j;
            
            if option.BatchSize==NN.numOfData
                Sample.Data{1}=data; Sample.Label{1}=label;
            else
                Sample=Shuffle(data,label,BatchSize);
            end
            
            for k=1:numel(Sample.Label)
                Counter=Counter+1;
                NN.StochasticCounter=Counter;
                ShuffledData=Sample.Data{k};
                ShuffledLabel=Sample.Label{k};
                if NN.WeightedFlag==1
                    NN.SampleWeight=NN.Weighted(Sample.Index{k});
                end
                [dw,db]=AutoGrad(ShuffledData,ShuffledLabel,NN);
                NN=StochasticUpdateRule(dw,db,NN,option);
                BatchCost(Counter)=CostFunction(ShuffledData,ShuffledLabel,NN);
                
            end

            CurrentCost=CostFunction(data,label,NN);
            if rem(j,floor(option.MaxIteration/20))==0
                FormatSpec = 'Iteration : %d , Cost : %16.8f \n';
                fprintf(FormatSpec,j,CurrentCost);
            end
            NN.OptimizationHistory(j)=CurrentCost;
        end

        NN.OptimizationTime=toc; NN.BatchCost=BatchCost;
        
        OptimizedNN=NN;
    end



    function OptimizedNN=QuasiNewtonSolver(data,label,NN,option)
        
        if isfield(NN,'TerminationContion')==0
            TerminationNorm=1e-5;
        else
            TerminationNorm=option.TerminateCondition;
        end

        
        NetworkType=NN.NetworkType;
        
        if isfield(option,'Damping')==0
            option.Damping='DoubleDamping';
        end

        if strcmp(NN.LineSearcher,'Off')==1
            option.Damping='DoubleDamping';
        end
        NN.Damping=option.Damping;
        % ------------- Select Gradient Solver----------------

        if isfield(option,'GradientSolver')==0
            switch NetworkType
                case 'ANN'
                    AutoGrad=@(data,label,NN) AutomaticGradient(data,label,NN);
                case'ResNet'
                    AutoGrad=@(data,label,NN) ElementWiseRNAG(data,label,NN);
            end
        else
            switch NetworkType
                case 'ANN'
                    GradientSolver=option.GradientSolver;
                    switch GradientSolver
                        case 'Element'
                            AutoGrad=@(data,label,NN) ElementWiseAG(data,label,NN);
                        case 'Column'
                            AutoGrad=@(data,label,NN) ColumnWiseAG(data,label,NN);
                        case 'General'
                            AutoGrad=@(data,label,NN) ComplexStepGradient(data,label,NN);
                    end
                case'ResNet'
                    GradientSolver=option.GradientSolver;
                    switch GradientSolver
                        case 'Element'
                            AutoGrad=@(data,label,NN) ElementWiseRNAG(data,label,NN);
                        case 'Column'
                            AutoGrad=@(data,label,NN) ColumnWiseRNAG(data,label,NN);
                        case 'General'
                            AutoGrad=@(data,label,NN) ComplexStepGradient(data,label,NN);
                    end
            end
        end
        % ------------- Select Gradient Solver----------------
        H0=speye(NN.numOfWeight+NN.numOfBias);
        [dwNew,dbNew]=AutoGrad(data,label,NN);
        H=H0;

        NN.Termination=0; NN.OptimizationFail=0;
        delta=1e-3;
        tic
        for m=1:option.MaxIteration
            
            NN.Iteration=m;
            NN=QuasiNewtonUpdate(NN);
            CurrentCost=NN.OptimizationHistory(m);
            if NN.Termination==1
                disp('------------------------------------------------------')
                FormatSpec = 'Reach Stop Criteria in %d Iterations, Cost :%16.8f\n';
                fprintf(FormatSpec,m,CurrentCost);
                fprintf('First Order Optimality : %8.7f\n',NN.FirstOrderOptimality)
                break
            end

            if NN.OptimizationFail==1
                disp('Opimization Fail');
                break
            end

            if rem(m,floor(option.MaxIteration/20))==0
                FormatSpec = 'Iteration : %d , Cost : %16.8f \n';
                fprintf(FormatSpec,m,CurrentCost);
            end

        end

        NN.OptimizationTime=toc;
        OptimizedNN=NN;

        function UpdatedNN=QuasiNewtonUpdate(NN)

            solver=option.Solver;
            switch solver
                
                case 'BFGS'

                    dw=dwNew; db=dbNew;
                    dwVec=LocalMtoV(dw);
                    dbVec=LocalMtoV(db);

                    weight0=LocalMtoV(NN.weight);
                    bias0=LocalMtoV(NN.bias);
                    p0=[weight0;bias0];
                    dp=[dwVec;dbVec];

                    if NN.Iteration==1 && strcmp(NN.LineSearcher,'Off')==0
                        dp=delta*dp;
                    end

                    % Quasi Newton Descent

                    SearchDirection=-H*dp;
                    NN.SearchDirection=SearchDirection;
                    SearchResults=LineSearch(SearchDirection,dp,data,label,NN);
                    if SearchResults.OptimalStep~=0
                        s0=SearchResults.OptimalStep;
                    else
                        s0=option.s0;
                    end
                    NN.OptimizationFail=SearchResults.Termination;
                    
                    NN.StepSizeHistory(m)=s0;
                    NN.LineSearchIteration(m)=SearchResults.Iteration;
                    NN.OptimizationHistory(m)=SearchResults.Cost;
                    s=s0*SearchDirection;
                    p0=p0+s;
                    
                    if NN.OptimizationFail==0

                        weight0=p0(1:NN.numOfWeight);
                        bias0=p0(NN.numOfWeight+1:end);
                        NN.weight=LocalVtoM(weight0);
                        NN.bias=LocalVtoM(bias0);
                        [dwNew,dbNew]=AutoGrad(data,label,NN);
                        dwNewVec=LocalMtoV(dwNew);
                        dbNewVec=LocalMtoV(dbNew);
                        dpNew=[dwNewVec;dbNewVec];
                        NN.Gradient=dpNew;

                        % BFGS Inverse Hessian Approximation Update
                        y=dpNew-dp;
                        rho=1/(y'*s);
                        NN.FirstOrderOptimality=max(abs(dpNew));
                        NN.rho(m)=rho;
                        % ------- Safegaurd -------
                        if NN.FirstOrderOptimality<=TerminationNorm
                            NN.Termination=1;
                        end

                        %-------------------------------------------------
                        if isfield(option,'Damping')==0
                            DampingCase='DoubleDamping';
                        else
                            DampingCase=option.Damping;
                        end
                        
                        switch DampingCase
                            case 'DoubleDamping'
                                % Quasi-Newton for DNN, Yi-Ren, Goldfarb 2022
                                mu1=0.2; mu2=0.001;

                                Quadratic=y'*H*y; InvRho=s'*y;
                                if InvRho<mu1*Quadratic
                                    theta=(1-mu1)*Quadratic/(Quadratic-InvRho);
                                    NN.CurvatureConditon(m)=0;
                                else
                                    theta=1;
                                    NN.CurvatureConditon(m)=1;
                                end
                                s=theta*s+(1-theta)*H*y;
                                
                                y=y+mu2*s;
                                %LM Damping
                                Rho=1/(s'*y);
                                H=H+(Rho^2)*(s'*y+y'*H*y)*(s*s')-Rho*(H*y*s'+s*y'*H);

                            case 'Powell'
                                % Quasi-Newton for DNN training, Goldfarb 2020  (Double Damping) 
                                % Powell's Damping on H, B=I.
                                mu1=0.2; mu2=0.001;

                                Quadratic=y'*H*y; InvRho=s'*y;
                                if InvRho<mu1*Quadratic
                                    theta=(1-mu1)*Quadratic/(Quadratic-InvRho);
                                    NN.CurvatureConditon(m)=0;
                                else
                                    theta=1;
                                    NN.CurvatureConditon(m)=1;
                                end
                                s=theta*s+(1-theta)*H*y;
                                NewInvRho=s'*y; Snorm=s'*s;
                                if NewInvRho<mu2*Snorm
                                    theta2=(1-mu2)*Snorm/(Snorm-NewInvRho);
                                else
                                    theta2=1;
                                end
                                
                                y=theta2*y+(1-theta2)*s;
                                Rho=1/(s'*y); InvRho=s'*y; Quadratic=y'*H*y;
                                if Quadratic*InvRho<=2/mu1
                                    H=H+(Rho^2)*(InvRho+Quadratic)*(s*s')-Rho*(H*y*s'+s*y'*H);
                                end
                            case 'Skip'
                                Rho=1/(s'*y); Quadratic=y'*H*y;
                                if rho>1e-8
                                    NN.CurvatureConditon(m)=1;
                                    H=H+(Rho^2)*(s'*y+Quadratic)*(s*s')-Rho*(H*y*s'+s*y'*H);
                                else
                                    NN.CurvatureConditon(m)=0;
                                end
                            case 'None'
                                Rho=1/(s'*y); Quadratic=y'*H*y;
                                H=H+(Rho^2)*(s'*y+Quadratic)*(s*s')-Rho*(H*y*s'+s*y'*H);
                        end
                        %-------------------------------------------------

                        NN.BFGS=H;
                        UpdatedNN=NN;
                    else
                        UpdatedNN=NN;
                    end
            end


            %%
            function ParaStruct=LocalVtoM(v)
                if numel(v)==NN.numOfWeight
                    NumOfVariable=0;

                    for i=1:NN.depth
                        NumOfLocalWeight=NN.LayerStruct(1,i)*NN.LayerStruct(2,i);
                        for j=1:NumOfLocalWeight
                            NumOfVariable=NumOfVariable+1;
                            NN.weight{i}(j)=v(NumOfVariable);
                        end
                    end
                    ParaStruct=NN.weight;
                else
                    NumOfVariable=0;
                    for i=1:NN.depth
                        NumOfLocalBias=NN.LayerStruct(2,i);
                        for j=1:NumOfLocalBias
                            NumOfVariable=NumOfVariable+1;
                            NN.bias{i}(j)=v(NumOfVariable);
                        end
                    end
                    ParaStruct=NN.bias;
                end
            end

            function Vector=LocalMtoV(S)
                VariableList=zeros(NN.depth,1);
                for i=1:NN.depth
                    VariableList(i)=numel(S{i});
                end
                TempVector=zeros(sum(VariableList),1);
                NumOfVariable=0;
                for i=1:NN.depth

                    for j=1:VariableList(i)
                        NumOfVariable=NumOfVariable+1;
                        TempVector(NumOfVariable)=S{i}(j);
                    end

                end
                Vector=TempVector;
            end

        end

    end

%------------------------------------------------------------------------------%
    function UpdatedNN=StochasticUpdateRule(dw,db,NN,option)

        solver=option.Solver;
        s0=option.s0;
        switch solver
        
            case "SGD"

                for j=1:NN.depth
                    NN.weight{j}=NN.weight{j}-s0*dw{j};
                    NN.bias{j}=NN.bias{j}-s0*db{j};
                end
                
            case "SGDM"
                
                m=0.9;
                for j=1:NN.depth
                    NN.FirstMomentW{j}=(m)*NN.FirstMomentW{j}+(1-m)*dw{j};
                    NN.FirstMomentB{j}=(m)*NN.FirstMomentB{j}+(1-m)*db{j};
                    NN.weight{j}=NN.weight{j}-s0*NN.FirstMomentW{j};
                    NN.bias{j}=NN.bias{j}-s0*NN.FirstMomentB{j};
                end
                
            case "RMSprop"

                for j=1:NN.depth

                    [DescentW,NN.FirstMomentW{j}]=RMSprop(dw{j},NN.FirstMomentW{j});
                    [DescentB,NN.FirstMomentB{j}]=RMSprop(db{j},NN.FirstMomentB{j});
                    NN.weight{j}=NN.weight{j}-s0*DescentW;
                    NN.bias{j}=NN.bias{j}-s0*DescentB;

                end
                
            case "ADAM"

                for j=1:NN.depth

                    [DescentW,FW,SW]=ADAM(dw{j},NN.FirstMomentW{j},NN.SecondMomentW{j});
                    [DescentB,FB,SB]=ADAM(db{j},NN.FirstMomentB{j},NN.SecondMomentB{j});

                    NN.FirstMomentW{j}=FW; NN.SecondMomentW{j}=SW;
                    NN.FirstMomentB{j}=FB; NN.SecondMomentB{j}=SB;
                    NN.weight{j}=NN.weight{j}-s0*DescentW;
                    NN.bias{j}=NN.bias{j}-s0*DescentB;
                    
                end
            case "AdamW"
                r=option.Regulator;
                for j=1:NN.depth

                    [DescentW,FW,SW]=ADAM(dw{j},NN.FirstMomentW{j},NN.SecondMomentW{j});
                    [DescentB,FB,SB]=ADAM(db{j},NN.FirstMomentB{j},NN.SecondMomentB{j});

                    NN.FirstMomentW{j}=FW; NN.SecondMomentW{j}=SW;
                    NN.FirstMomentB{j}=FB; NN.SecondMomentB{j}=SB;

                    NN.weight{j}=NN.weight{j}-s0*(DescentW+r*NN.weight{j});
                    NN.bias{j}=NN.bias{j}-s0*(DescentB+r*NN.bias{j});

                end

        end
        UpdatedNN=NN;

        function [d,Mnew,Vnew]=ADAM(dw,Mprev,Vprev)
            iter=NN.StochasticCounter;
            beta1=0.9; beta2=0.999;
            Mnew=(beta1)*Mprev+(1-beta1)*dw;
            Vnew=(beta2)*Vprev+(1-beta2)*(dw.^2);
            Mt=Mnew/(1-beta1^iter); Vt=Vnew/(1-beta2^iter);
            epsilon=(1e-8);
            d=Mt./(sqrt(Vt)+epsilon);
        end
        
        function [d,Vnew]=RMSprop(dw,Vprev)
            beta=0.9;
            Vnew=(beta)*Vprev+(1-beta)*(dw.^2);
            epsilon=(1e-8);
            d=dw./(sqrt(Vnew)+epsilon);
        end

    end

end
%% Auxiliary Function
function Sample=Shuffle(data,label,BatchSize)
NumOfData=numel(data(1,:));
NumOfBatch=floor(NumOfData/BatchSize)+1;
LastBatch=rem(NumOfData,BatchSize);
Index=randperm(NumOfData);
for i=1:NumOfBatch
    if i~=NumOfBatch
        Rand=Index((i-1)*BatchSize+1:i*BatchSize);
        Sample.Data{i}=data(:,Rand);
        Sample.Label{i}=label(:,Rand);
        Sample.Index{i}=Rand;
    elseif i==NumOfBatch && LastBatch~=0
        Rand=Index(NumOfData-LastBatch+1:end);
        Sample.Data{i}=data(:,Rand);
        Sample.Label{i}=label(:,Rand);
        Sample.Index{i}=Rand;
    end

end
end

function Accuracy=ComputeAccuracy(data,label,NN)
    Probability=NN.Evaluate(data);
    [~,PredictIndex]=max(Probability);
    LabelIndex=NN.HotToIndex(label);
    CorrectVector=LabelIndex==PredictIndex;
    Accuracy=100*mean(CorrectVector);

end

function Class=ClassPredict(data,NN)
    Probability=NN.Evaluate(data);
    [~,Class]=max(Probability);
end
