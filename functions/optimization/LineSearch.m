function SearchResults = LineSearch(SearchDirection, Gradient, data, label, NN)
% LineSearch
% IO unchanged: SearchResults = LineSearch(SearchDirection, Gradient, data, label, NN)

    if strcmp(NN.LineSearcher,'Off') == 1
        SearchResults.Termination = 0;
        SearchResults.Cost        = CostFunction(data,label,NN);
        SearchResults.OptimalStep = 0;
        SearchResults.Iteration   = 0;
        return
    end

    % ------------------------------------------------------------
    % Line-search / descent-related configuration (centralized)
    % ------------------------------------------------------------
    cfg = getLineSearchConfig(NN);

    % ------------------------------------------------------------
    % Build structured direction
    % ------------------------------------------------------------
    Direction = buildDirectionFromVector(SearchDirection, NN);

    % ------------------------------------------------------------
    % Initialize interval + interpolation state
    % ------------------------------------------------------------
    Interval = cfg.intervalDefault;
    interp   = initInterpolation(Interval, Direction, data, label, NN, Gradient, SearchDirection);

    C0 = interp.Cost0;
    D0 = interp.Derivate0;

    wolfe = checkStrongWolfe(C0, D0, interp.Step1, interp.Cost1, interp.Derivate1, cfg);
    compromise = 0;

    if wolfe.isStrong == 1
        SearchResults.Termination = 0;
        SearchResults.OptimalStep = interp.Step1;
        SearchResults.Iteration   = 0;
        SearchResults.Cost        = C0;  % kept as original behavior
        return
    end

    % ------------------------------------------------------------
    % Cubic interpolation loop + compromise logic (algorithm unchanged)
    % ------------------------------------------------------------
    Candidate      = zeros(cfg.LSMaxIter, 2);
    Candidate(:,2) = Inf(cfg.LSMaxIter, 1);

    Existence = 1;

    for i = 1:cfg.LSMaxIter
        Estimate = CubicInterpolation(interp);

        if ~isValidEstimate(Estimate, cfg)
            Existence = 0;
            break
        end

        CostC     = DirectionalCost(Estimate, Direction, data, label, NN);
        DerivateC = DirectionalDerivate(Estimate, Direction, data, label, NN);

        wolfe = checkStrongWolfe(C0, D0, Estimate, CostC, DerivateC, cfg);

        if wolfe.isStrong == 1
            step = Estimate;
            break
        elseif (wolfe.w1 == 1) && (wolfe.w2 == 0)
            Candidate(i,1) = Estimate;
            Candidate(i,2) = CostC;
        end

        hasCandidate = any(~isinf(Candidate(:,2)));

        if (sum(Candidate(:,1)) == 0) && (i == cfg.LSMaxIter)
            Existence = 0;
        end

        if (wolfe.isStrong == 0) && (i == cfg.LSMaxIter) && (hasCandidate == 1)
            [~, idx] = min(Candidate(:,2));
            step = Candidate(idx, 1);
            compromise = 4;
        elseif (hasCandidate ~= 1) && (i == cfg.LSMaxIter)
            step = Estimate;
        end

        if (wolfe.isStrong == 0) && (i == cfg.LSMaxIter)
            Existence = 0;
        end

        [Interval, interp] = updateInterpolation(Interval, interp, Estimate, CostC, DerivateC);
    end

    % ------------------------------------------------------------
    % Fallback: backtracking
    % ------------------------------------------------------------
    if Existence == 0
        Searcher = NN.LineSearcher;
        switch Searcher
            case 'BackTrack'
                SearchResults = BackTracking(C0, D0, cfg.BTMaxIterBackTrack, cfg, Direction, data, label, NN);
            case 'Iterative'
                SearchResults = BackTracking(C0, D0, cfg.BTMaxIterIterative, cfg, Direction, data, label, NN);
                SearchResults.Termination = 0;
        end
        return
    end

    % ------------------------------------------------------------
    % Success
    % ------------------------------------------------------------
    SearchResults.Termination = 0;
    SearchResults.Cost        = CostC;
    SearchResults.OptimalStep = step;

    if compromise == 0
        SearchResults.Iteration = i;
    else
        SearchResults.Iteration = compromise;
    end
end

% ======================================================================
% Config (centralize ALL "descent/line-search" knobs here)
% ======================================================================

function cfg = getLineSearchConfig(NN) %#ok<INUSD>
    % Cubic-interpolation search
    cfg.LSMaxIter       = 3;
    cfg.intervalDefault = [0, 5];

    % Strong Wolfe parameters
    cfg.c1 = 1e-4;
    cfg.c2 = 0.9;

    % Safeguards for cubic interpolation estimate
    cfg.imagTol    = 1e-30;
    cfg.stepAbsTol = 1e-8;

    % Backtracking parameters (kept behavior: c1 fixed here too)
    cfg.BTDecayRate          = 0.5;
    cfg.BTInitialStep        = 1;
    cfg.BTMaxIterBackTrack   = 30;
    cfg.BTMaxIterIterative   = 10;

    % Note: if you later want to tune per NN.LineSearcher mode, do it here.
end

% ======================================================================
% Helpers
% ======================================================================

function Direction = buildDirectionFromVector(SearchDirection, NN)
    wDir = SearchDirection(1:NN.numOfWeight);
    bDir = SearchDirection(NN.numOfWeight+1:NN.numOfWeight+NN.numOfBias);

    wDir = VecToMatrix(wDir, NN);
    bDir = VecToMatrix(bDir, NN);

    Direction.Weight = wDir;
    Direction.Bias   = bDir;

    if isfield(NN,'splineOn') && NN.splineOn
        cDir = SearchDirection(NN.numOfWeight+NN.numOfBias+1:end);
        nK = NN.bsplineGrid.numBasis;
        nL = NN.depth - 1;
        Direction.Spline = cell(nL, 1);
        offset = 0;
        for ii = 1:nL
            Direction.Spline{ii} = cDir(offset+1:offset+nK);
            offset = offset + nK;
        end
    end
end

function interp = initInterpolation(Interval, Direction, data, label, NN, Gradient, SearchDirection)
    interp.Step0 = Interval(1);
    interp.Step1 = Interval(2);

    interp.Derivate0 = Gradient' * SearchDirection;
    interp.Derivate1 = DirectionalDerivate(interp.Step1, Direction, data, label, NN);

    interp.Cost0 = DirectionalCost(interp.Step0, Direction, data, label, NN);
    interp.Cost1 = DirectionalCost(interp.Step1, Direction, data, label, NN);
end

function ok = isValidEstimate(Estimate, cfg)
    if isnan(Estimate)
        ok = false; return
    end
    if abs(imag(Estimate)) >= cfg.imagTol
        ok = false; return
    end
    if abs(Estimate) <= cfg.stepAbsTol
        ok = false; return
    end
    ok = true;
end

function wolfe = checkStrongWolfe(C0, D0, step, Cstep, Dstep, cfg)
% Returns:
%   wolfe.w1       : Armijo (sufficient decrease)
%   wolfe.w2       : curvature
%   wolfe.isStrong : strong Wolfe (w1 & w2)

    wolfe.w1 = (Cstep <= (C0 + cfg.c1 * D0 * step));
    wolfe.w2 = (abs(Dstep) <= (-cfg.c2 * D0));
    wolfe.isStrong = (wolfe.w1 == 1 && wolfe.w2 == 1);
end

function [Interval, interp] = updateInterpolation(Interval, interp, Estimate, CostC, DerivateC)
    Interval = sort(Interval);

    if DerivateC > 0
        Interval(2)      = Estimate;
        interp.Step1     = Interval(2);
        interp.Cost1     = CostC;
        interp.Derivate1 = DerivateC;
    else
        Interval(1)      = Estimate;
        interp.Step0     = Interval(1);
        interp.Cost0     = CostC;
        interp.Derivate0 = DerivateC;
    end
end

% ======================================================================
% Backtracking (signature kept via wrapper usage)
% ======================================================================

function SearchResults = BackTracking(C0, D0, BTMaxIter, cfg, Direction, data, label, NN)
    StepB     = cfg.BTInitialStep;
    c1        = cfg.c1;            % keep Armijo constant consistent
    DecayRate = cfg.BTDecayRate;

    CostRecord = zeros(BTMaxIter,1);

    for i = 1:BTMaxIter
        Wolfe1LHS      = DirectionalCost(StepB, Direction, data, label, NN);
        CostRecord(i)  = Wolfe1LHS;

        Wolfe1RHS      = C0 + c1 * D0 * StepB;
        Wolfe1OK       = (Wolfe1LHS <= Wolfe1RHS);

        if Wolfe1OK == 1
            Step = StepB;
            break
        elseif Wolfe1OK == 0 && i == BTMaxIter
            Step = StepB;
        else
            StepB = DecayRate * StepB;
        end
    end

    [minimu, ~] = min(CostRecord);
    Fail = (minimu >= C0);

    if Fail == 0
        SearchResults.Termination = 0;
        SearchResults.Cost        = Wolfe1LHS;
        SearchResults.OptimalStep = Step;
        SearchResults.Iteration   = -i;
    else
        SearchResults.Termination = 0;
        SearchResults.Cost        = C0;
        SearchResults.OptimalStep = Step;
        SearchResults.Iteration   = -50;
    end
end

% ======================================================================
% Original local functions (kept signatures)
% ======================================================================

function Output = DirectionalCost(Step, Direction, data, label, NN)
    for j = 1:NN.depth
        NN.weight{j} = NN.weight{j} + (Step) * Direction.Weight{j};
        NN.bias{j}   = NN.bias{j}   + (Step) * Direction.Bias{j};
    end
    if isfield(Direction,'Spline')
        for j = 1:NN.depth-1
            NN.splineCoeff{j} = NN.splineCoeff{j} + Step * Direction.Spline{j};
        end
    end
    Output = CostFunction(data,label,NN);
end

function EstimateStep = CubicInterpolation(Object)
    fv0   = Object.Cost0;      fv1   = Object.Cost1;
    dv0   = Object.Derivate0;  dv1   = Object.Derivate1;
    step0 = Object.Step0;      step1 = Object.Step1;

    d1 = dv0 + dv1 - 3*(fv0 - fv1)/(step0 - step1);
    d2 = sign(step1 - step0) * sqrt(d1^2 - dv0*dv1);

    EstimateStep = step1 - (step1 - step0) * (dv1 + d2 - d1) / (dv1 - dv0 + 2*d2);
end