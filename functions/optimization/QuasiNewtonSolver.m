function OptimizedNN = QuasiNewtonSolver(data,label,NN,option)
%QUASINEWTONSOLVER  Quasi-Newton or (L-)BFGS optimization.
%   Smooth wrapper that encapsulates both full BFGS and limited-memory LBFGS
%   implementations. Many helper routines are kept as subfunctions to keep the
%   top-level flow readable.

% termination norm
if isfield(option,'TerminationContion')
    TerminationNorm = option.TerminateCondition;
else
    TerminationNorm = 1e-5;
end

% ---- LBFGS defaults ----
if ~isfield(option, 'lbfgsMemory')
    option.lbfgsMemory = 10;          % typical: 5~20
end
if ~isfield(option, 'lbfgsYTsMin')
    option.lbfgsYTsMin = 1e-12;       % curvature safeguard
end

% initialize lbfgs memory inside NN if needed
if ~isfield(NN, 'lbfgsState')
    NN.lbfgsState.sList   = {};
    NN.lbfgsState.yList   = {};
    NN.lbfgsState.rhoList = [];
    NN.lbfgsState.gamma   = 1.0;      % H0 = gamma * I
    NN.lbfgsState.maxMem  = option.lbfgsMemory;
else
    NN.lbfgsState.maxMem  = option.lbfgsMemory;
end

NetworkType = NN.NetworkType;

if ~isfield(option,'Damping')
    option.Damping = 'DoubleDamping';
end
if strcmp(NN.LineSearcher,'Off')
    option.Damping = 'DoubleDamping';
end
NN.Damping = option.Damping;

% choose gradient solver (same logic as in original file)
if ~isfield(option,'GradientSolver')
    switch NetworkType
        case 'ANN'
            AutoGrad=@(d,l,nn) AutomaticGradient(d,l,nn);
        case 'ResNet'
            AutoGrad=@(d,l,nn) ResNetAutomaticGradient(d,l,nn);
        case 'SirenNet'
            AutoGrad=@(d,l,nn) SirenNetAutomaticGradient(d,l,nn);
    end
else
    GradientSolver = option.GradientSolver;
    switch NetworkType
        case 'ANN'
            switch GradientSolver
                case 'Element'
                    AutoGrad=@(d,l,nn) ElementWiseAG(d,l,nn);
                case 'Column'
                    AutoGrad=@(d,l,nn) ColumnWiseAG(d,l,nn);
                case 'General'
                    AutoGrad=@(d,l,nn) ComplexStepGradient(d,l,nn);
            end
        case 'ResNet'
            switch GradientSolver
                case 'Element'
                    AutoGrad=@(d,l,nn) ElementWiseRNAG(d,l,nn);
                case 'Column'
                    AutoGrad=@(d,l,nn) ColumnWiseRNAG(d,l,nn);
                case 'General'
                    AutoGrad=@(d,l,nn) ComplexStepGradient(d,l,nn);
            end
        case 'SirenNet'
            AutoGrad=@(d,l,nn) ComplexStepGradient(d,l,nn);
    end
end

% initial quadratic approx for BFGS
splineOn = isfield(NN,'splineOn') && NN.splineOn;
H0 = speye(NN.numOfWeight+NN.numOfBias+NN.numOfSpline);
[dwNew,dbNew,dcNew] = AutoGrad(data,label,NN);
H = H0;

NN.Termination = 0; NN.OptimizationFail = 0;
delta = 1e-3;
tic
for m = 1:option.MaxIteration
    NN.Iteration = m;
    NN = QuasiNewtonUpdate(NN);
    CurrentCost = NN.OptimizationHistory(m);
    if NN.Termination == 1
        disp('------------------------------------------------------')
        FormatSpec = 'Reach Stop Criteria in %d Iterations, Cost :%16.8f\n';
        fprintf(FormatSpec,m,CurrentCost);
        fprintf('First Order Optimality : %8.7f\n',NN.FirstOrderOptimality)
        break
    end
    if NN.OptimizationFail == 1
        disp('Optimization Fail');
        break
    end
    if rem(m,floor(option.MaxIteration/20)) == 0
        FormatSpec = 'Iteration : %d , Cost : %16.8f \n';
        fprintf(FormatSpec,m,CurrentCost);
    end
end

NN.OptimizationTime = toc;
OptimizedNN = NN;

%% subfunctions ------------------------------------------------------------
    function updatedNN = QuasiNewtonUpdate(NN)
        solver = option.Solver;
        % note: some option defaults repeated to keep independence
        if ~isfield(option,'lbfgsMemory')
            option.lbfgsMemory = 10;
        end
        if ~isfield(option,'lbfgsYTsMin')
            option.lbfgsYTsMin = 1e-12;
        end
        if ~isfield(NN,'lbfgsState')
            NN.lbfgsState.sList   = {};
            NN.lbfgsState.yList   = {};
            NN.lbfgsState.rhoList = [];
            NN.lbfgsState.gamma   = 1.0;
            NN.lbfgsState.maxMem  = option.lbfgsMemory;
        else
            NN.lbfgsState.maxMem  = option.lbfgsMemory;
        end

        switch solver
            case 'BFGS'
                %------------- original BFGS branch --------------------------------
                dw = dwNew; db = dbNew; dc_loc = dcNew;
                dwVec = LocalMtoV(dw);
                dbVec = LocalMtoV(db);
                weight0 = LocalMtoV(NN.weight);
                bias0   = LocalMtoV(NN.bias);
                if splineOn
                    dcVec    = SplineCellToVec(dc_loc);
                    spline0  = SplineCellToVec(NN.splineCoeff);
                    p0       = [weight0; bias0; spline0];
                    dp       = [dwVec; dbVec; dcVec];
                else
                    p0       = [weight0; bias0];
                    dp       = [dwVec; dbVec];
                end
                if NN.Iteration==1 && strcmp(NN.LineSearcher,'Off')==0
                    dp = delta * dp;
                end
                searchDirection = -H * dp;
                NN.SearchDirection = searchDirection;
                searchResults = LineSearch(searchDirection, dp, data, label, NN);
                if searchResults.OptimalStep~=0
                    stepSize = searchResults.OptimalStep;
                else
                    stepSize = option.s0;
                end
                NN.OptimizationFail = searchResults.Termination;
                NN.StepSizeHistory(m)     = stepSize;
                NN.LineSearchIteration(m) = searchResults.Iteration;
                NN.OptimizationHistory(m) = searchResults.Cost;
                sVec = stepSize * searchDirection;
                p0   = p0 + sVec;
                if NN.OptimizationFail==0
                    weight0 = p0(1:NN.numOfWeight);
                    bias0   = p0(NN.numOfWeight+1:NN.numOfWeight+NN.numOfBias);
                    NN.weight = LocalVtoM(weight0);
                    NN.bias   = LocalVtoM(bias0);
                    if splineOn
                        spline0 = p0(NN.numOfWeight+NN.numOfBias+1:end);
                        NN.splineCoeff = SplineVecToCell(spline0, NN);
                    end
                    [dwNew,dbNew,dcNew] = AutoGrad(data,label,NN);
                    dwNewVec = LocalMtoV(dwNew);
                    dbNewVec = LocalMtoV(dbNew);
                    if splineOn
                        dcNewVec = SplineCellToVec(dcNew);
                        dpNew    = [dwNewVec; dbNewVec; dcNewVec];
                    else
                        dpNew    = [dwNewVec; dbNewVec];
                    end
                    NN.Gradient = dpNew;
                    yVec = dpNew - dp;
                    rho  = 1/(yVec' * sVec);
                    NN.FirstOrderOptimality = max(abs(dpNew));
                    NN.rho(m) = rho;
                    if NN.FirstOrderOptimality <= TerminationNorm
                        NN.Termination = 1;
                    end
                    if ~isfield(option,'Damping')
                        dampingCase = 'DoubleDamping';
                    else
                        dampingCase = option.Damping;
                    end
                    switch dampingCase
                        case 'DoubleDamping'
                            mu1 = 0.2; mu2 = 0.001;
                            quadratic = yVec' * H * yVec;
                            invRho    = sVec' * yVec;
                            if invRho < mu1 * quadratic
                                theta = (1-mu1) * quadratic / (quadratic - invRho);
                                NN.CurvatureConditon(m) = 0;
                            else
                                theta = 1;
                                NN.CurvatureConditon(m) = 1;
                            end
                            sVec = theta*sVec + (1-theta)*H*yVec;
                            yVec = yVec + mu2*sVec;
                            Rho = 1/(sVec' * yVec);
                            H = H + (Rho^2) * (sVec'*yVec + yVec'*H*yVec) * (sVec*sVec') ...
                                  - Rho * (H*yVec*sVec' + sVec*yVec'*H);
                        case 'Powell'
                            mu1 = 0.2; mu2 = 0.001;
                            quadratic = yVec' * H * yVec;
                            invRho    = sVec' * yVec;
                            if invRho < mu1 * quadratic
                                theta = (1-mu1) * quadratic / (quadratic - invRho);
                                NN.CurvatureConditon(m) = 0;
                            else
                                theta = 1;
                                NN.CurvatureConditon(m) = 1;
                            end
                            sVec = theta*sVec + (1-theta)*H*yVec;
                            newInvRho = sVec' * yVec;
                            sNorm     = sVec' * sVec;
                            if newInvRho < mu2 * sNorm
                                theta2 = (1-mu2) * sNorm / (sNorm - newInvRho);
                            else
                                theta2 = 1;
                            end
                            yVec = theta2*yVec + (1-theta2)*sVec;
                            Rho = 1/(sVec' * yVec);
                            invRho = sVec' * yVec;
                            quadratic = yVec' * H * yVec;
                            if quadratic * invRho <= 2/mu1
                                H = H + (Rho^2) * (invRho + quadratic) * (sVec*sVec') ...
                                      - Rho * (H*yVec*sVec' + sVec*yVec'*H);
                            end
                        case 'Skip'
                            Rho = 1/(sVec' * yVec);
                            if rho > 1e-8
                                NN.CurvatureConditon(m) = 1;
                                quadratic = yVec' * H * yVec;
                                H = H + (Rho^2) * (sVec'*yVec + quadratic) * (sVec*sVec') ...
                                      - Rho * (H*yVec*sVec' + sVec*yVec'*H);
                            else
                                NN.CurvatureConditon(m) = 0;
                            end
                        case 'None'
                            Rho = 1/(sVec' * yVec);
                            quadratic = yVec' * H * yVec;
                            H = H + (Rho^2) * (sVec'*yVec + quadratic) * (sVec*sVec') ...
                                  - Rho * (H*yVec*sVec' + sVec*yVec'*H);
                    end
                    NN.BFGS = H;
                    updatedNN = NN;
                else
                    updatedNN = NN;
                end

            case 'LBFGS'
                % limited-memory variant
                dw = dwNew; db = dbNew; dc_loc = dcNew;
                dwVec = LocalMtoV(dw);
                dbVec = LocalMtoV(db);
                weight0 = LocalMtoV(NN.weight);
                bias0   = LocalMtoV(NN.bias);
                if splineOn
                    dcVec    = SplineCellToVec(dc_loc);
                    spline0  = SplineCellToVec(NN.splineCoeff);
                    p0       = [weight0; bias0; spline0];
                    dp       = [dwVec; dbVec; dcVec];
                else
                    p0       = [weight0; bias0];
                    dp       = [dwVec; dbVec];
                end
                NN.Gradient = dp;
                if NN.Iteration==1 && strcmp(NN.LineSearcher,'Off')==0
                    dp = delta * dp;
                end
                invHg = applyLbfgsInvH(dp, NN.lbfgsState);
                searchDirection = -invHg;
                NN.SearchDirection = searchDirection;
                searchResults = LineSearch(searchDirection, dp, data, label, NN);
                if searchResults.OptimalStep~=0
                    stepSize = searchResults.OptimalStep;
                else
                    stepSize = option.s0;
                end
                NN.OptimizationFail = searchResults.Termination;
                NN.StepSizeHistory(m)     = stepSize;
                NN.LineSearchIteration(m) = searchResults.Iteration;
                NN.OptimizationHistory(m) = searchResults.Cost;
                sVec = stepSize * searchDirection;
                p1   = p0 + sVec;
                if NN.OptimizationFail==0
                    weight1 = p1(1:NN.numOfWeight);
                    bias1   = p1(NN.numOfWeight+1:NN.numOfWeight+NN.numOfBias);
                    NN.weight = LocalVtoM(weight1);
                    NN.bias   = LocalVtoM(bias1);
                    if splineOn
                        spline1 = p1(NN.numOfWeight+NN.numOfBias+1:end);
                        NN.splineCoeff = SplineVecToCell(spline1, NN);
                    end
                    [dwNew,dbNew,dcNew] = AutoGrad(data,label,NN);
                    dwNewVec = LocalMtoV(dwNew);
                    dbNewVec = LocalMtoV(dbNew);
                    if splineOn
                        dcNewVec = SplineCellToVec(dcNew);
                        dpNew    = [dwNewVec; dbNewVec; dcNewVec];
                    else
                        dpNew    = [dwNewVec; dbNewVec];
                    end
                    NN.Gradient = dpNew;
                    NN.FirstOrderOptimality = max(abs(dpNew));
                    if NN.FirstOrderOptimality <= TerminationNorm
                        NN.Termination = 1;
                    end
                    yVec = dpNew - dp;
                    % double damping
                    mu1 = 0.2;
                    mu2 = 0.001;
                    Hy = applyLbfgsInvH(yVec, NN.lbfgsState);
                    quadratic = yVec' * Hy;
                    invRho    = sVec' * yVec;
                    if invRho < mu1 * quadratic
                        theta = (1-mu1) * quadratic / (quadratic - invRho);
                        NN.CurvatureConditon(m) = 0;
                    else
                        theta = 1;
                        NN.CurvatureConditon(m) = 1;
                    end
                    sVec = theta * sVec + (1-theta) * Hy;
                    yVec = yVec + mu2 * sVec;
                    yTs = yVec' * sVec;
                    % update memory
                    if yTs > option.lbfgsYTsMin
                        yTy = yVec' * yVec;
                        if yTy > 0
                            NN.lbfgsState.gamma = yTs / yTy;
                        else
                            NN.lbfgsState.gamma = 1.0;
                        end
                        NN.lbfgsState = pushLbfgsPair(NN.lbfgsState, sVec, yVec, yTs);
                    else
                        NN.CurvatureConditon(m) = 0;
                    end
                    updatedNN = NN;
                else
                    updatedNN = NN;
                end
        end
    end

    function invHg = applyLbfgsInvH(gVec, lbfgsState)
        gVec = gVec(:);
        n = numel(gVec);
        k = numel(lbfgsState.rhoList);
        if k == 0
            invHg = lbfgsState.gamma * gVec;
            return
        end
        stale = false;
        if numel(lbfgsState.sList) ~= k || numel(lbfgsState.yList) ~= k
            stale = true;
        else
            for ii = 1:k
                if numel(lbfgsState.sList{ii}) ~= n || numel(lbfgsState.yList{ii}) ~= n
                    stale = true;
                    break
                end
            end
        end
        if stale
            lbfgsState.sList   = {};
            lbfgsState.yList   = {};
            lbfgsState.rhoList = [];
            lbfgsState.gamma   = 1.0;
            invHg = lbfgsState.gamma * gVec;
            return
        end
        alpha = zeros(k, 1);
        qVec = gVec;
        for i = k:-1:1
            sI = lbfgsState.sList{i}; sI = sI(:);
            yI = lbfgsState.yList{i}; yI = yI(:);
            rhoI = lbfgsState.rhoList(i);
            alpha(i) = rhoI * (sI' * qVec);
            qVec = qVec - alpha(i) * yI;
        end
        rVec = lbfgsState.gamma * qVec;
        for i = 1:k
            sI = lbfgsState.sList{i}; sI = sI(:);
            yI = lbfgsState.yList{i}; yI = yI(:);
            rhoI = lbfgsState.rhoList(i);
            beta = rhoI * (yI' * rVec);
            rVec = rVec + sI * (alpha(i) - beta);
        end
        invHg = rVec;
    end

    function lbfgsState = pushLbfgsPair(lbfgsState, sVec, yVec, yTs)
        sVec = sVec(:);
        yVec = yVec(:);
        rho = 1.0 / yTs;
        lbfgsState.sList{end+1} = sVec;
        lbfgsState.yList{end+1} = yVec;
        lbfgsState.rhoList(end+1,1) = rho;
        if numel(lbfgsState.rhoList) > lbfgsState.maxMem
            keepStart = numel(lbfgsState.rhoList) - lbfgsState.maxMem + 1;
            lbfgsState.sList   = lbfgsState.sList(keepStart:end);
            lbfgsState.yList   = lbfgsState.yList(keepStart:end);
            lbfgsState.rhoList = lbfgsState.rhoList(keepStart:end);
        end
    end

    function ParaStruct = LocalVtoM(v)
        if numel(v) == NN.numOfWeight
            NumOfVariable = 0;
            for i = 1:NN.depth
                NumOfLocalWeight = NN.LayerStruct(1,i) * NN.LayerStruct(2,i);
                for j = 1:NumOfLocalWeight
                    NumOfVariable = NumOfVariable + 1;
                    NN.weight{i}(j) = v(NumOfVariable);
                end
            end
            ParaStruct = NN.weight;
        else
            NumOfVariable = 0;
            for i = 1:NN.depth
                NumOfLocalBias = NN.LayerStruct(2,i);
                for j = 1:NumOfLocalBias
                    NumOfVariable = NumOfVariable + 1;
                    NN.bias{i}(j) = v(NumOfVariable);
                end
            end
            ParaStruct = NN.bias;
        end
    end

    function Vector = LocalMtoV(S)
        VariableList = zeros(NN.depth,1);
        for i = 1:NN.depth
            VariableList(i) = numel(S{i});
        end
        TempVector = zeros(sum(VariableList),1);
        NumOfVariable = 0;
        for i = 1:NN.depth
            for j = 1:VariableList(i)
                NumOfVariable = NumOfVariable + 1;
                TempVector(NumOfVariable) = S{i}(j);
            end
        end
        Vector = TempVector;
    end

    function vec = SplineCellToVec(cells)
        vec = vertcat(cells{:});
    end

    function cells = SplineVecToCell(v, NN_)
        if isfield(NN_,'bsplineOn') && NN_.bsplineOn
            nK = NN_.bsplineGrid.numBasis;
        else
            nK = NN_.splineGrid.numKnots;
        end
        nL = NN_.depth - 1;
        cells = cell(nL, 1);
        offset = 0;
        for i = 1:nL
            cells{i} = v(offset+1:offset+nK);
            offset = offset + nK;
        end
    end
end