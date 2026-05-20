function optimizedNN = QuasiNewtonSolver(data, label, NN, option)
%QUASINEWTONSOLVER  BFGS / L-BFGS optimization with analytical gradients.
%
% The main loop is intentionally small:
%   1. pack parameters and gradient
%   2. compute a quasi-Newton search direction
%   3. line-search along that direction
%   4. update parameters and curvature state

    option = normalizeOptions(option, NN);
    autoGrad = selectAnalyticalGradient(NN, option);
    splineOn = isfield(NN, 'splineOn') && NN.splineOn;
    lnOn = isfield(NN, 'layerNormOn') && NN.layerNormOn;

    NN.Damping = option.Damping;
    NN.Termination = 0;
    NN.OptimizationFail = 0;
    NN.FirstOrderOptimality = Inf;

    paramVec = packParameters(NN, splineOn, lnOn);
    state = initQuasiNewtonState(option, numel(paramVec), NN);

    [dw, db, dc, dlnG, dlnB] = autoGrad(data, label, NN);
    gradVec = packGradient(dw, db, dc, NN, splineOn, lnOn, dlnG, dlnB);

    history = initHistory(option.MaxIteration, option.storeHistory);
    progressInterval = max(1, floor(option.MaxIteration / 20));
    lastCost = CostFunction(data, label, NN);
    lastIteration = 0;

    tic
    for iter = 1:option.MaxIteration
        lastIteration = iter;
        NN.Iteration = iter;

        lineGradient = gradVec;
        if iter == 1 && ~strcmp(NN.LineSearcher, 'Off')
            lineGradient = option.firstStepGradientScale * lineGradient;
        end

        searchDirection = computeSearchDirection(lineGradient, state, option);
        searchResults = LineSearch(searchDirection, lineGradient, data, label, NN);
        stepSize = chooseStepSize(searchResults, option);
        lastCost = searchResults.Cost;

        NN.OptimizationFail = searchResults.Termination;
        history = recordLineSearch(history, iter, stepSize, searchResults);

        if NN.OptimizationFail == 1
            disp('Optimization Fail');
            break
        end

        stepVec = stepSize * searchDirection;
        paramVec = paramVec + stepVec;
        NN = unpackParameters(paramVec, NN, splineOn, lnOn);

        [dw, db, dc, dlnG, dlnB] = autoGrad(data, label, NN);
        newGradVec = packGradient(dw, db, dc, NN, splineOn, lnOn, dlnG, dlnB);

        yVec = newGradVec - lineGradient;
        NN.FirstOrderOptimality = max(abs(newGradVec));
        NN.Termination = double(NN.FirstOrderOptimality <= option.terminationNorm);

        [state, curvatureOk] = updateQuasiNewtonState(state, stepVec, yVec, option);
        history = recordCurvature(history, iter, curvatureOk);

        gradVec = newGradVec;
        history = recordCost(history, iter, lastCost);

        if NN.Termination == 1
            disp('------------------------------------------------------')
            fprintf('Reach Stop Criteria in %d Iterations, Cost :%16.8f\n', iter, lastCost);
            fprintf('First Order Optimality : %8.7f\n', NN.FirstOrderOptimality)
            break
        end

        if rem(iter, progressInterval) == 0
            fprintf('Iteration : %d , Cost : %16.8f \n', iter, lastCost);
        end
    end

    NN.Iteration = lastIteration;
    NN.OptimizationTime = toc;

    if option.storeHistory
        NN.OptimizationHistory = history.cost(1:lastIteration);
        NN.StepSizeHistory = history.stepSize(1:lastIteration);
        NN.LineSearchIteration = history.lineSearchIteration(1:lastIteration);
        NN.CurvatureConditon = history.curvature(1:lastIteration);
        if strcmp(option.Solver, 'LBFGS')
            NN.lbfgsState = state.lbfgs;
        end
    end

    if option.storeBfgs && strcmp(option.Solver, 'BFGS')
        NN.BFGS = state.H;
    end

    optimizedNN = NN;
end

function option = normalizeOptions(option, NN)
    if ~isfield(option, 'Solver')
        option.Solver = 'BFGS';
    end
    option.Solver = char(option.Solver);
    if ~any(strcmp(option.Solver, {'BFGS', 'LBFGS'}))
        error('QuasiNewtonSolver:UnsupportedSolver', 'Solver must be BFGS or LBFGS.');
    end

    if ~isfield(option, 's0')
        option.s0 = 2e-3;
    end
    if ~isfield(option, 'MaxIteration')
        option.MaxIteration = 500;
    end
    if isfield(option, 'TerminateCondition')
        option.terminationNorm = option.TerminateCondition;
    elseif isfield(option, 'TerminationContion')
        option.terminationNorm = option.TerminationContion;
    else
        option.terminationNorm = 1e-5;
    end
    if ~isfield(option, 'Damping')
        option.Damping = 'DoubleDamping';
    end
    if strcmp(NN.LineSearcher, 'Off')
        option.Damping = 'DoubleDamping';
    end
    if ~isfield(option, 'lbfgsMemory')
        option.lbfgsMemory = 10;
    end
    if ~isfield(option, 'lbfgsYTsMin')
        option.lbfgsYTsMin = 1e-12;
    end
    if ~isfield(option, 'storeHistory')
        option.storeHistory = false;
    end
    if ~isfield(option, 'storeBfgs')
        option.storeBfgs = false;
    end
    option.firstStepGradientScale = 1e-3;
end

function autoGrad = selectAnalyticalGradient(NN, option)
    if isfield(option, 'GradientSolver') && ~strcmpi(option.GradientSolver, 'Analytical')
        error('QuasiNewtonSolver:UnsupportedGradientSolver', ...
            ['Numerical full-gradient solvers were removed from training. ', ...
             'Use the default analytical gradient solver.']);
    end

    switch NN.NetworkType
        case 'ANN'
            autoGrad = @(d,l,nn) AutomaticGradient(d,l,nn);
        case 'ResNet'
            autoGrad = @(d,l,nn) ResNetAutomaticGradient(d,l,nn);
        otherwise
            error('QuasiNewtonSolver:UnsupportedNetworkType', ...
                'NetworkType must be ANN or ResNet.');
    end
end

function state = initQuasiNewtonState(option, numParameters, NN)
    state = struct();
    if strcmp(option.Solver, 'BFGS')
        state.H = speye(numParameters);
    else
        if isfield(NN, 'lbfgsState')
            state.lbfgs = NN.lbfgsState;
            state.lbfgs.maxMem = option.lbfgsMemory;
        else
            state.lbfgs = emptyLbfgsState(option.lbfgsMemory);
        end
    end
end

function lbfgs = emptyLbfgsState(maxMem)
    lbfgs.sList = {};
    lbfgs.yList = {};
    lbfgs.rhoList = [];
    lbfgs.gamma = 1.0;
    lbfgs.maxMem = maxMem;
end

function history = initHistory(maxIteration, storeHistory)
    history.store = storeHistory;
    if storeHistory
        history.cost = zeros(maxIteration, 1);
        history.stepSize = zeros(maxIteration, 1);
        history.lineSearchIteration = zeros(maxIteration, 1);
        history.curvature = zeros(maxIteration, 1);
    else
        history.cost = [];
        history.stepSize = [];
        history.lineSearchIteration = [];
        history.curvature = [];
    end
end

function searchDirection = computeSearchDirection(gradientVec, state, option)
    if strcmp(option.Solver, 'BFGS')
        searchDirection = -state.H * gradientVec;
    else
        searchDirection = -applyLbfgsInvH(gradientVec, state.lbfgs);
    end
end

function stepSize = chooseStepSize(searchResults, option)
    if searchResults.OptimalStep ~= 0
        stepSize = searchResults.OptimalStep;
    else
        stepSize = option.s0;
    end
end

function [state, curvatureOk] = updateQuasiNewtonState(state, sVec, yVec, option)
    if strcmp(option.Solver, 'BFGS')
        [state.H, curvatureOk] = updateBfgsInverseH(state.H, sVec, yVec, option.Damping);
    else
        [state.lbfgs, curvatureOk] = updateLbfgsState(state.lbfgs, sVec, yVec, option);
    end
end

function [H, curvatureOk] = updateBfgsInverseH(H, sVec, yVec, dampingCase)
    curvatureOk = 1;

    switch dampingCase
        case 'DoubleDamping'
            mu1 = 0.2; mu2 = 0.001;
            quadratic = yVec' * H * yVec;
            invRho = sVec' * yVec;
            if invRho < mu1 * quadratic
                theta = (1 - mu1) * quadratic / (quadratic - invRho);
                curvatureOk = 0;
            else
                theta = 1;
            end
            sVec = theta * sVec + (1 - theta) * H * yVec;
            yVec = yVec + mu2 * sVec;
            H = inverseBfgsFormula(H, sVec, yVec);

        case 'Powell'
            mu1 = 0.2; mu2 = 0.001;
            quadratic = yVec' * H * yVec;
            invRho = sVec' * yVec;
            if invRho < mu1 * quadratic
                theta = (1 - mu1) * quadratic / (quadratic - invRho);
                curvatureOk = 0;
            else
                theta = 1;
            end
            sVec = theta * sVec + (1 - theta) * H * yVec;
            newInvRho = sVec' * yVec;
            sNorm = sVec' * sVec;
            if newInvRho < mu2 * sNorm
                theta2 = (1 - mu2) * sNorm / (sNorm - newInvRho);
            else
                theta2 = 1;
            end
            yVec = theta2 * yVec + (1 - theta2) * sVec;
            invRho = sVec' * yVec;
            quadratic = yVec' * H * yVec;
            if quadratic * invRho <= 2 / mu1
                H = inverseBfgsFormula(H, sVec, yVec);
            else
                curvatureOk = 0;
            end

        case 'Skip'
            if sVec' * yVec > 1e-8
                H = inverseBfgsFormula(H, sVec, yVec);
            else
                curvatureOk = 0;
            end

        case 'None'
            H = inverseBfgsFormula(H, sVec, yVec);

        otherwise
            error('QuasiNewtonSolver:UnknownDamping', ...
                'Unknown damping option "%s".', dampingCase);
    end
end

function H = inverseBfgsFormula(H, sVec, yVec)
    yTs = yVec' * sVec;
    if abs(yTs) <= eps
        return
    end
    rho = 1 / yTs;
    quadratic = yVec' * H * yVec;
    H = H + (rho^2) * (yTs + quadratic) * (sVec * sVec') ...
          - rho * (H * yVec * sVec' + sVec * yVec' * H);
end

function [lbfgs, curvatureOk] = updateLbfgsState(lbfgs, sVec, yVec, option)
    curvatureOk = 1;
    mu1 = 0.2;
    mu2 = 0.001;

    Hy = applyLbfgsInvH(yVec, lbfgs);
    quadratic = yVec' * Hy;
    invRho = sVec' * yVec;
    if invRho < mu1 * quadratic
        theta = (1 - mu1) * quadratic / (quadratic - invRho);
        curvatureOk = 0;
    else
        theta = 1;
    end

    sVec = theta * sVec + (1 - theta) * Hy;
    yVec = yVec + mu2 * sVec;
    yTs = yVec' * sVec;
    if yTs <= option.lbfgsYTsMin
        curvatureOk = 0;
        return
    end

    yTy = yVec' * yVec;
    if yTy > 0
        lbfgs.gamma = yTs / yTy;
    else
        lbfgs.gamma = 1.0;
    end
    lbfgs = pushLbfgsPair(lbfgs, sVec, yVec, yTs);
end

function invHg = applyLbfgsInvH(gVec, lbfgs)
    gVec = gVec(:);
    n = numel(gVec);
    k = numel(lbfgs.rhoList);
    if k == 0
        invHg = lbfgs.gamma * gVec;
        return
    end

    if isStaleLbfgsMemory(lbfgs, k, n)
        invHg = gVec;
        return
    end

    alpha = zeros(k, 1);
    qVec = gVec;
    for idx = k:-1:1
        sI = lbfgs.sList{idx}(:);
        yI = lbfgs.yList{idx}(:);
        rhoI = lbfgs.rhoList(idx);
        alpha(idx) = rhoI * (sI' * qVec);
        qVec = qVec - alpha(idx) * yI;
    end

    rVec = lbfgs.gamma * qVec;
    for idx = 1:k
        sI = lbfgs.sList{idx}(:);
        yI = lbfgs.yList{idx}(:);
        rhoI = lbfgs.rhoList(idx);
        beta = rhoI * (yI' * rVec);
        rVec = rVec + sI * (alpha(idx) - beta);
    end
    invHg = rVec;
end

function stale = isStaleLbfgsMemory(lbfgs, k, n)
    stale = numel(lbfgs.sList) ~= k || numel(lbfgs.yList) ~= k;
    if stale
        return
    end
    for idx = 1:k
        if numel(lbfgs.sList{idx}) ~= n || numel(lbfgs.yList{idx}) ~= n
            stale = true;
            return
        end
    end
end

function lbfgs = pushLbfgsPair(lbfgs, sVec, yVec, yTs)
    lbfgs.sList{end + 1} = sVec(:);
    lbfgs.yList{end + 1} = yVec(:);
    lbfgs.rhoList(end + 1, 1) = 1 / yTs;
    if numel(lbfgs.rhoList) > lbfgs.maxMem
        keepStart = numel(lbfgs.rhoList) - lbfgs.maxMem + 1;
        lbfgs.sList = lbfgs.sList(keepStart:end);
        lbfgs.yList = lbfgs.yList(keepStart:end);
        lbfgs.rhoList = lbfgs.rhoList(keepStart:end);
    end
end

function paramVec = packParameters(NN, splineOn, lnOn)
    paramVec = [MatrixToVec(NN.weight, NN); MatrixToVec(NN.bias, NN)];
    if splineOn
        paramVec = [paramVec; splineCellToVec(NN.splineCoeff)];
    end
    if lnOn
        paramVec = [paramVec; lnCellToVec(NN.lnGamma); lnCellToVec(NN.lnBeta)];
    end
end

function gradVec = packGradient(dw, db, dc, NN, splineOn, lnOn, dlnG, dlnB)
    gradVec = [MatrixToVec(dw, NN); MatrixToVec(db, NN)];
    if splineOn
        if ~iscell(dc)
            error('QuasiNewtonSolver:MissingSplineGradient', ...
                'Spline networks require coefficient gradients.');
        end
        gradVec = [gradVec; splineCellToVec(dc)];
    end
    if lnOn
        gradVec = [gradVec; lnCellToVec(dlnG); lnCellToVec(dlnB)];
    end
end

function NN = unpackParameters(paramVec, NN, splineOn, lnOn)
    weightEnd = NN.numOfWeight;
    biasEnd = NN.numOfWeight + NN.numOfBias;
    NN.weight = VecToMatrix(paramVec(1:weightEnd), NN);
    NN.bias = VecToMatrix(paramVec(weightEnd + 1:biasEnd), NN);
    offset = biasEnd;
    if splineOn
        splineEnd = offset + NN.numOfSpline;
        NN.splineCoeff = splineVecToCell(paramVec(offset + 1:splineEnd), NN);
        offset = splineEnd;
    end
    if lnOn
        halfLN = NN.numOfLN / 2;
        NN.lnGamma = lnVecToCell(paramVec(offset + 1:offset + halfLN), NN);
        offset = offset + halfLN;
        NN.lnBeta = lnVecToCell(paramVec(offset + 1:offset + halfLN), NN);
    end
end

function vec = splineCellToVec(cells)
    vec = vertcat(cells{:});
end

function cells = splineVecToCell(vec, NN)
    numCoeff = NN.bsplineGrid.numBasis;
    numLayers = NN.depth - 1;
    cells = cell(numLayers, 1);
    offset = 0;
    for idx = 1:numLayers
        cells{idx} = vec(offset + 1:offset + numCoeff);
        offset = offset + numCoeff;
    end
end

function vec = lnCellToVec(cells)
    vec = vertcat(cells{:});
end

function cells = lnVecToCell(vec, NN)
    numLayers = NN.depth - 1;
    cells = cell(numLayers, 1);
    offset = 0;
    for idx = 1:numLayers
        H = size(NN.weight{idx}, 1);
        cells{idx} = vec(offset + 1:offset + H);
        offset = offset + H;
    end
end

function history = recordLineSearch(history, iter, stepSize, searchResults)
    if ~history.store
        return
    end
    history.stepSize(iter) = stepSize;
    history.lineSearchIteration(iter) = searchResults.Iteration;
end

function history = recordCost(history, iter, costValue)
    if history.store
        history.cost(iter) = costValue;
    end
end

function history = recordCurvature(history, iter, curvatureOk)
    if history.store
        history.curvature(iter) = curvatureOk;
    end
end
