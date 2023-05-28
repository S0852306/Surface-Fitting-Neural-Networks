function FunctionOutput=ResNet(data,NN)

if strcmp(NN.InputAutoScaling,'on')==1
    data=NN.InputScaleVector.*data-NN.InputCenterVector;
end
v=data;

for i=1:NN.depth-1 
    z=NN.weight{i}*v+NN.bias{i};

    if i>1
        v=NN.active(z)+NN.ResMap{i}*vp;
    else
        v=NN.active(z);

    end
    vp=z;
end
v=NN.OutActive(NN.weight{NN.depth}*v+NN.bias{NN.depth});


FunctionOutput=v;