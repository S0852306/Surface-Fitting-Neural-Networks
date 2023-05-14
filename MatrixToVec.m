function Vector=MatrixToVec(S,NN)
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

