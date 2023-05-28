function SearchResults=LineSearch(SearchDirection,Gradient,data,label,NN)

if strcmp(NN.LineSearcher,'Off')==1
    SearchResults.Termination=0;
    SearchResults.Cost=CostFunction(data,label,NN);
    SearchResults.OptimalStep=0;
    SearchResults.Iteration=0;
    return
end


%% Dimension Adjustment
WeightSearchDirection=SearchDirection(1:NN.numOfWeight);
BiasSearchDirection=SearchDirection(NN.numOfWeight+1:end);
WeightSearchDirection=VecToMatrix(WeightSearchDirection,NN);
BiasSearchDirection=VecToMatrix(BiasSearchDirection,NN);
Direction.Weight=WeightSearchDirection;
Direction.Bias=BiasSearchDirection;


%%
LSMaxIter=3;
IntervalDefault=[0,10];
% Existence=0;
Interval=IntervalDefault;


Interpolation.Step0=Interval(1);
Interpolation.Step1=Interval(2);
Derivate0=Gradient'*SearchDirection;
Derivate1=DirectionalDerivate(Interval(2),Direction,data,label,NN);
Interpolation.Cost0=DirectionalCost(Interval(1),Direction,data,label,NN);
Interpolation.Cost1=DirectionalCost(Interval(2),Direction,data,label,NN);
Interpolation.Derivate0=Derivate0;
Interpolation.Derivate1=Derivate1;
C0=Interpolation.Cost0;
D0=Derivate0;

c1=1e-4; c2=0.9;
Wolfe1LHS=Interpolation.Cost1;
Wolfe1RHS=Interpolation.Cost0+c1*D0*Interval(2);
Wolfe1Condition=Wolfe1LHS<=Wolfe1RHS;
Wolfe2Condition=abs(Derivate1)<=-c2*D0;
StrongWolfeCondition=(Wolfe1Condition==1 && Wolfe2Condition==1);
compromise=0;
if StrongWolfeCondition==1
    SearchResults.Termination=0;
    step=Interval(2);
    SearchResults.OptimalStep=step;
    SearchResults.Iteration=0;
    SearchResults.Cost=C0;

else
    Candidate=zeros(LSMaxIter,2);
    Candidate(:,2)=Inf(LSMaxIter,1);

    for i=1:LSMaxIter
        
        Estimate=CubicInterpolation(Interpolation);
        %-------------Safegaurd-----------------------------
        if abs(imag(Estimate))>=1e-30 
            Existence=0;
            break
        else
            Existence=1;
        end


        if abs(Estimate)<=1e-8
            Existence=0;
            break
        end

        if isnan(Estimate)
            Existence=0;
            break
        else
            Existence=1;
        end
        %---------------Safegaurd End----------------------

        CostC=DirectionalCost(Estimate,Direction,data,label,NN);

        % Verify Wolfe Condition
        Wolfe1LHS=CostC; Wolfe1RHS=C0+c1*D0*Estimate;
        Wolfe1Condition=Wolfe1LHS<=Wolfe1RHS;
        DerivateC=DirectionalDerivate(Estimate,Direction,data,label,NN);
        Wolfe2Condition=abs(DerivateC)<=-c2*D0;
        StrongWolfeCondition=(Wolfe1Condition==1 && Wolfe2Condition==1);
        if StrongWolfeCondition==1
            step=Estimate;
            break
        elseif Wolfe1Condition==1 && Wolfe2Condition==0
            Candidate(i,1)=Estimate;
            Candidate(i,2)=CostC;
        end
        
        CandidateCondition=isinf(Candidate(:,2));
        CandidateCondition=sum(CandidateCondition)~=LSMaxIter;
        
        if sum(Candidate(:,1))==0 && i==LSMaxIter
            Existence=0;
        end

        if StrongWolfeCondition==0 && i==LSMaxIter && CandidateCondition==1
            [~,index]=min(Candidate(:,2));
            step=Candidate(index,1);
            compromise=4;
        elseif CandidateCondition~=1 && i==LSMaxIter
            step=Estimate;
        end
        

        
        if StrongWolfeCondition==0 && i==LSMaxIter
            Existence=0;
        end
        % Elimination & Assign New Interval for Cubic Interpolation
        Interval=sort(Interval);

        if DerivateC>0
            Interval(2)=Estimate;
            Interpolation.Step1=Interval(2);
            Interpolation.Cost1=CostC;
            Interpolation.Derivate1=DerivateC;
        else
            Interval(1)=Estimate;
            Interpolation.Step0=Interval(1);
            Interpolation.Cost0=CostC;
            Interpolation.Derivate0=DerivateC;
        end

    end
    
    if Existence==0
        % Perform Simple Back Tracking Line Search
        Searcher=NN.LineSearcher;
        switch Searcher
            case 'BackTrack'
                SearchResults=BackTracking(C0,D0,30,Direction,data,label,NN);
            case 'Iterative'
                SearchResults=BackTracking(C0,D0,10,Direction,data,label,NN);
                SearchResults.Termination=0;
            
                
        end
        % Negative iterations imply using Back Tracking
    else
        SearchResults.Termination=0;
        SearchResults.Cost=CostC;
        SearchResults.OptimalStep=step;
        if compromise==0
            SearchResults.Termination=0;
            SearchResults.Iteration=i;
        else
            SearchResults.Termination=0;
            SearchResults.Iteration=compromise;
        end

    end
    

    
end


end

function SearchResults=BackTracking(C0,D0,BTMaxIter,Direction,data,label,NN)
InitialStep=1; StepB=InitialStep;
c1=1e-4; 
% DecayRate=0.618033988749895;
DecayRate=0.5;
for i=1:BTMaxIter
    Wolfe1LHS=DirectionalCost(StepB,Direction,data,label,NN);
    CostRecord(i)=Wolfe1LHS;
    Wolfe1RHS=C0+c1*D0*StepB;
    Wolfe1Condition=Wolfe1LHS<=Wolfe1RHS;
    if Wolfe1Condition==1
        Step=StepB;
        break
    elseif Wolfe1Condition==0 && i==BTMaxIter
        Step=StepB;
    else
        StepB=DecayRate*StepB;
    end

end
[minimu,BTindex]=min(CostRecord);
Fail=minimu>=C0;
if Fail==0
    SearchResults.Termination=0;
    SearchResults.Cost=Wolfe1LHS;
    SearchResults.OptimalStep=Step;
    SearchResults.Iteration=-i;

elseif Fail==0 && Wolfe1Condition==0
    SearchResults.Termination=0;
    SearchResults.Cost=C0;
    SearchResults.OptimalStep=IntervalBackTracking(2)*DecayRate^(BTindex-1);
    SearchResults.Iteration=-50;

else
    SearchResults.Termination=0;
    SearchResults.Cost=C0;
    SearchResults.OptimalStep=Step;
    SearchResults.Iteration=-50;
end

end

function Output=DirectionalCost(Step,Direction,data,label,NN)

for j=1:NN.depth

    NN.weight{j}=NN.weight{j}+(Step)*Direction.Weight{j};
    NN.bias{j}=NN.bias{j}+(Step)*Direction.Bias{j};
end

Output=CostFunction(data,label,NN);

end



function EstimateStep=CubicInterpolation(Object)
fv0=Object.Cost0; fv1=Object.Cost1;
dv0=Object.Derivate0; dv1=Object.Derivate1;
step0=Object.Step0; step1=Object.Step1;
d1=dv0+dv1-3*(fv0-fv1)/(step0-step1);
d2=sign(step1-step0)*sqrt(d1^2-dv0*dv1);
EstimateStep=step1-(step1-step0)*(dv1+d2-d1)/(dv1-dv0+2*d2);
end