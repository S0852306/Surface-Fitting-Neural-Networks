function FunctionOutput=ANN(data,NN)
v=data;

for i=1:NN.depth-1 
    v=NN.weight{i}*v+NN.bias{i};

    v=NN.active(v);
end
v=NN.OutActive(NN.weight{NN.depth}*v+NN.bias{NN.depth});
FunctionOutput=v;