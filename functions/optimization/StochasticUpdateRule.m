function UpdatedNN = StochasticUpdateRule(dw,db,NN,option,dc)
%STOCHASTICUPDATERULE  Apply parameter update based on chosen optimizer.
%   UpdatedNN = StochasticUpdateRule(dw,db,NN,option) modifies the network
%   parameters using the solver specified in option.Solver. Supported
%   solvers include "SGD", "SGDM", "RMSprop", "ADAM" and "AdamW".
%   Optional 5th argument dc: cell array of spline coefficient gradients.

if nargin < 5; dc = {}; end
splineOn = isfield(NN,'splineOn') && NN.splineOn && ~isempty(dc);

solver = option.Solver;
s0 = option.s0;
switch solver
    case "SGD"
        for j = 1:NN.depth
            NN.weight{j} = NN.weight{j} - s0 * dw{j};
            NN.bias{j}   = NN.bias{j}   - s0 * db{j};
        end
        if splineOn
            for j = 1:NN.depth-1
                NN.splineCoeff{j} = NN.splineCoeff{j} - s0 * dc{j};
            end
        end

    case "SGDM"
        m = 0.9;
        for j = 1:NN.depth
            NN.FirstMomentW{j} = m * NN.FirstMomentW{j} + (1-m) * dw{j};
            NN.FirstMomentB{j} = m * NN.FirstMomentB{j} + (1-m) * db{j};
            NN.weight{j} = NN.weight{j} - s0 * NN.FirstMomentW{j};
            NN.bias{j}   = NN.bias{j} - s0 * NN.FirstMomentB{j};
        end
        if splineOn
            for j = 1:NN.depth-1
                NN.FirstMomentC{j} = m * NN.FirstMomentC{j} + (1-m) * dc{j};
                NN.splineCoeff{j} = NN.splineCoeff{j} - s0 * NN.FirstMomentC{j};
            end
        end

    case "RMSprop"
        for j = 1:NN.depth
            [DescentW,NN.FirstMomentW{j}] = RMSprop(dw{j},NN.FirstMomentW{j});
            [DescentB,NN.FirstMomentB{j}] = RMSprop(db{j},NN.FirstMomentB{j});
            NN.weight{j} = NN.weight{j} - s0 * DescentW;
            NN.bias{j}   = NN.bias{j} - s0 * DescentB;
        end
        if splineOn
            for j = 1:NN.depth-1
                [DescentC,NN.FirstMomentC{j}] = RMSprop(dc{j},NN.FirstMomentC{j});
                NN.splineCoeff{j} = NN.splineCoeff{j} - s0 * DescentC;
            end
        end

    case "ADAM"
        for j = 1:NN.depth
            [DescentW,FW,SW] = ADAM(dw{j},NN.FirstMomentW{j},NN.SecondMomentW{j});
            [DescentB,FB,SB] = ADAM(db{j},NN.FirstMomentB{j},NN.SecondMomentB{j});

            NN.FirstMomentW{j}  = FW; NN.SecondMomentW{j}  = SW;
            NN.FirstMomentB{j}  = FB; NN.SecondMomentB{j}  = SB;
            NN.weight{j} = NN.weight{j} - s0 * DescentW;
            NN.bias{j}   = NN.bias{j} - s0 * DescentB;
        end
        if splineOn
            for j = 1:NN.depth-1
                [DescentC,FC,SC] = ADAM(dc{j},NN.FirstMomentC{j},NN.SecondMomentC{j});
                NN.FirstMomentC{j} = FC; NN.SecondMomentC{j} = SC;
                NN.splineCoeff{j} = NN.splineCoeff{j} - s0 * DescentC;
            end
        end

    case "AdamW"
        r = option.Regulator;
        for j = 1:NN.depth
            [DescentW,FW,SW] = ADAM(dw{j},NN.FirstMomentW{j},NN.SecondMomentW{j});
            [DescentB,FB,SB] = ADAM(db{j},NN.FirstMomentB{j},NN.SecondMomentB{j});

            NN.FirstMomentW{j} = FW; NN.SecondMomentW{j} = SW;
            NN.FirstMomentB{j} = FB; NN.SecondMomentB{j} = SB;

            NN.weight{j} = NN.weight{j} - s0 * (DescentW + r * NN.weight{j});
            NN.bias{j}   = NN.bias{j}   - s0 * (DescentB + r * NN.bias{j});
        end
        if splineOn
            for j = 1:NN.depth-1
                [DescentC,FC,SC] = ADAM(dc{j},NN.FirstMomentC{j},NN.SecondMomentC{j});
                NN.FirstMomentC{j} = FC; NN.SecondMomentC{j} = SC;
                NN.splineCoeff{j} = NN.splineCoeff{j} - s0 * (DescentC + r * NN.splineCoeff{j});
            end
        end
end

UpdatedNN = NN;

%% subfunctions for optimizers
    function [d,Mnew,Vnew] = ADAM(dw,Mprev,Vprev)
        iter = NN.StochasticCounter;
        beta1 = 0.9; beta2 = 0.999;
        Mnew = beta1 * Mprev + (1-beta1) * dw;
        Vnew = beta2 * Vprev + (1-beta2) * (dw.^2);
        Mt = Mnew / (1 - beta1^iter);
        Vt = Vnew / (1 - beta2^iter);
        epsilon = 1e-8;
        d = Mt ./ (sqrt(Vt) + epsilon);
    end

    function [d,Vnew] = RMSprop(dw,Vprev)
        beta = 0.9;
        Vnew = beta * Vprev + (1-beta) * (dw.^2);
        epsilon = 1e-8;
        d = dw ./ (sqrt(Vnew) + epsilon);
    end
end