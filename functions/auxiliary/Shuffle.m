function Sample = Shuffle(data,label,BatchSize)
%SHUFFLE  Split dataset into randomly shuffled mini-batches.
%   Sample = Shuffle(data,label,BatchSize) returns a struct with fields
%   Data, Label, and Index. Each cell corresponds to one mini-batch.

NumOfData = size(data,2);
NumOfBatch = floor(NumOfData/BatchSize) + 1;
LastBatch = rem(NumOfData,BatchSize);
Index = randperm(NumOfData);

Sample.Data  = cell(NumOfBatch,1);
Sample.Label = cell(NumOfBatch,1);
Sample.Index = cell(NumOfBatch,1);

for i = 1:NumOfBatch
    if i ~= NumOfBatch
        Rand = Index((i-1)*BatchSize+1 : i*BatchSize);
    else
        Rand = Index(NumOfData-LastBatch+1 : end);
    end
    Sample.Data{i}  = data(:,Rand);
    Sample.Label{i} = label(:,Rand);
    Sample.Index{i} = Rand;
end
end