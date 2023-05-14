function ParaStruct=VecToMatrix(v,NN)


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