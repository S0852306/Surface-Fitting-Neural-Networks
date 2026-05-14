function updatedNN = StochasticUpdateRule(dw, db, NN, option, dc)
%STOCHASTICUPDATERULE  Apply one stochastic optimizer step.

    if nargin < 5
        dc = {};
    end

    solver = char(option.Solver);
    stepSize = option.s0;
    splineOn = isfield(NN, 'splineOn') && NN.splineOn && ~isempty(dc);

    switch solver
        case 'SGD'
            NN.weight = applySgd(NN.weight, dw, stepSize);
            NN.bias = applySgd(NN.bias, db, stepSize);
            if splineOn
                NN.splineCoeff = applySgd(NN.splineCoeff, dc, stepSize);
            end

        case 'SGDM'
            momentum = 0.9;
            [NN.weight, NN.FirstMomentW] = applyMomentum(NN.weight, dw, NN.FirstMomentW, stepSize, momentum);
            [NN.bias, NN.FirstMomentB] = applyMomentum(NN.bias, db, NN.FirstMomentB, stepSize, momentum);
            if splineOn
                [NN.splineCoeff, NN.FirstMomentC] = applyMomentum(NN.splineCoeff, dc, NN.FirstMomentC, stepSize, momentum);
            end

        case 'RMSprop'
            [NN.weight, NN.FirstMomentW] = applyRmsprop(NN.weight, dw, NN.FirstMomentW, stepSize);
            [NN.bias, NN.FirstMomentB] = applyRmsprop(NN.bias, db, NN.FirstMomentB, stepSize);
            if splineOn
                [NN.splineCoeff, NN.FirstMomentC] = applyRmsprop(NN.splineCoeff, dc, NN.FirstMomentC, stepSize);
            end

        case 'ADAM'
            iter = NN.StochasticCounter;
            [NN.weight, NN.FirstMomentW, NN.SecondMomentW] = applyAdam(NN.weight, dw, NN.FirstMomentW, NN.SecondMomentW, stepSize, iter);
            [NN.bias, NN.FirstMomentB, NN.SecondMomentB] = applyAdam(NN.bias, db, NN.FirstMomentB, NN.SecondMomentB, stepSize, iter);
            if splineOn
                [NN.splineCoeff, NN.FirstMomentC, NN.SecondMomentC] = applyAdam(NN.splineCoeff, dc, NN.FirstMomentC, NN.SecondMomentC, stepSize, iter);
            end

        case 'AdamW'
            iter = NN.StochasticCounter;
            regulator = getRegulator(option);
            [NN.weight, NN.FirstMomentW, NN.SecondMomentW] = applyAdamW(NN.weight, dw, NN.FirstMomentW, NN.SecondMomentW, stepSize, iter, regulator);
            [NN.bias, NN.FirstMomentB, NN.SecondMomentB] = applyAdamW(NN.bias, db, NN.FirstMomentB, NN.SecondMomentB, stepSize, iter, regulator);
            if splineOn
                [NN.splineCoeff, NN.FirstMomentC, NN.SecondMomentC] = applyAdamW(NN.splineCoeff, dc, NN.FirstMomentC, NN.SecondMomentC, stepSize, iter, regulator);
            end

        otherwise
            error('StochasticUpdateRule:UnknownSolver', ...
                'Unknown stochastic solver "%s".', solver);
    end

    updatedNN = NN;
end

function params = applySgd(params, grads, stepSize)
    for idx = 1:numel(params)
        params{idx} = params{idx} - stepSize * grads{idx};
    end
end

function [params, momentumState] = applyMomentum(params, grads, momentumState, stepSize, momentum)
    for idx = 1:numel(params)
        momentumState{idx} = momentum * momentumState{idx} + (1 - momentum) * grads{idx};
        params{idx} = params{idx} - stepSize * momentumState{idx};
    end
end

function [params, rmsState] = applyRmsprop(params, grads, rmsState, stepSize)
    for idx = 1:numel(params)
        [direction, rmsState{idx}] = rmspropDirection(grads{idx}, rmsState{idx});
        params{idx} = params{idx} - stepSize * direction;
    end
end

function [params, firstMoment, secondMoment] = applyAdam(params, grads, firstMoment, secondMoment, stepSize, iter)
    for idx = 1:numel(params)
        [direction, firstMoment{idx}, secondMoment{idx}] = adamDirection(grads{idx}, firstMoment{idx}, secondMoment{idx}, iter);
        params{idx} = params{idx} - stepSize * direction;
    end
end

function [params, firstMoment, secondMoment] = applyAdamW(params, grads, firstMoment, secondMoment, stepSize, iter, regulator)
    for idx = 1:numel(params)
        [direction, firstMoment{idx}, secondMoment{idx}] = adamDirection(grads{idx}, firstMoment{idx}, secondMoment{idx}, iter);
        params{idx} = params{idx} - stepSize * (direction + regulator * params{idx});
    end
end

function regulator = getRegulator(option)
    if isfield(option, 'Regulator')
        regulator = option.Regulator;
    else
        regulator = 0;
    end
end

function [direction, firstMoment, secondMoment] = adamDirection(grad, firstMoment, secondMoment, iter)
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-8;

    firstMoment = beta1 * firstMoment + (1 - beta1) * grad;
    secondMoment = beta2 * secondMoment + (1 - beta2) * (grad.^2);

    correctedFirst = firstMoment / (1 - beta1^iter);
    correctedSecond = secondMoment / (1 - beta2^iter);
    direction = correctedFirst ./ (sqrt(correctedSecond) + epsilon);
end

function [direction, rmsState] = rmspropDirection(grad, rmsState)
    beta = 0.9;
    epsilon = 1e-8;
    rmsState = beta * rmsState + (1 - beta) * (grad.^2);
    direction = grad ./ (sqrt(rmsState) + epsilon);
end
