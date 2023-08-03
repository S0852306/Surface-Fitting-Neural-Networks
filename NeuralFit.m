function FittedModel=NeuralFit(data,label,DimensionVector)
InputDimension=DimensionVector(1); 
OutputDimension=DimensionVector(2);

LayerStruct=[InputDimension,10,10,10,OutputDimension];

if size(data,1)~=InputDimension
    data=data.';
end

if size(label,1)~=OutputDimension
    label=label.';
end
NumOfData=size(label,2);
NN.InputAutoScaling='on';
NN.LabelAutoScaling='on';
NN=Initialization(LayerStruct,NN);
option.Solver='ADAM';
if NumOfData>=200
    option.BatchSize=floor(NumOfData/15);
else
    option.BatchSize=floor(NumOfData/5);
end
option.s0=2e-3;
option.MaxIteration=50;
disp('------------------------------------------------------')
DisplayWord=['Optimization will terminate after ', num2str(600) ' Iterations.'];
disp(DisplayWord)
disp('------------------------------------------------------')
NN=OptimizationSolver(data,label,NN,option);

DisplayWord=['First Stage Optimization Finished in  ', num2str(option.MaxIteration), '  Iterations.'];
disp(DisplayWord)
disp('------------------------------------------------------')

option.Solver='BFGS';
option.MaxIteration=550;
NN=OptimizationSolver(data,label,NN,option);
NN.Report=FittingReport(data,label,NN);
FittedModel=NN;