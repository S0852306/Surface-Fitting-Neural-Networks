function Report = FittingReport(data, label, NN, opts)
% FittingReport
% Compute regression metrics and (optionally) visualize:
%   1) Predict vs Actual
%   2) Error distribution
%
% Backward compatible:
%   Report = FittingReport(data, label, NN);
%
% Options (all optional):
%   opts.plot             (default true)
%   opts.closeFigures     (default false)
%   opts.numHistBins      (default 30)
%   opts.markerSize       (default 18)
%
%   opts.print            (default true)   % print metrics
%   opts.storeVectors     (default 'auto') % 'auto'|'all'|'none'|'sample'
%   opts.maxStoreElements (default 2e6)    % threshold for auto store (dout*N)
%   opts.sampleStoreN     (default 20000)  % when storeVectors='sample'
%   opts.maxPlotPoints    (default 5000)   % downsample scatter if N is huge
%   opts.randomSeed       (default 1)      % used for sampling consistency
%
% Notes:
% - data : (din x N)
% - label: (dout x N)
% - NN must provide: prediction = NN.Evaluate(data)

    if nargin < 4 || isempty(opts)
        opts = struct();
    end

    opts = applyDefaults(opts, struct( ...
        'plot', true, ...
        'closeFigures', false, ...
        'numHistBins', 30, ...
        'markerSize', 18, ...
        'print', true, ...
        'storeVectors', 'auto', ...     % 'auto'|'all'|'none'|'sample'
        'maxStoreElements', 2e6, ...    % elements = dout*N
        'sampleStoreN', 20000, ...
        'maxPlotPoints', 5000, ...
        'randomSeed', 1 ...
    ));

    if opts.closeFigures
        close all;
    end

    % ----------------------------
    % Prediction
    % ----------------------------
    prediction = NN.Evaluate(data);

    if size(label,1) ~= size(prediction,1) || size(label,2) ~= size(prediction,2)
        error('FittingReport:SizeMismatch', 'label and prediction must have same size.');
    end

    nOut = size(label,1);
    N    = size(label,2);

    % ----------------------------
    % Errors
    % ----------------------------
    err = label - prediction;
    sqe = err.^2;

    muY = mean(label,2);
    SST = sum((label - muY).^2,2);
    SSE = sum(sqe,2);

    MSE  = mean(sqe,2);
    RMSE = sqrt(MSE);
    MAE  = mean(abs(err),2);
    Bias = mean(err,2);

    % Robust R^2
    Rsq = zeros(nOut,1);
    for i = 1:nOut
        if SST(i) < 1e-12
            if SSE(i) < 1e-12
                Rsq(i) = 1;
            else
                Rsq(i) = 0;
            end
        else
            Rsq(i) = 1 - SSE(i)/SST(i);
        end
    end

    % ----------------------------
    % Decide whether to store big vectors
    % ----------------------------
    storeMode = lower(opts.storeVectors);
    totalElements = nOut * N;

    doStoreAll = false;
    doStoreNone = false;
    doStoreSample = false;

    if strcmp(storeMode,'all')
        doStoreAll = true;
    elseif strcmp(storeMode,'none')
        doStoreNone = true;
    elseif strcmp(storeMode,'sample')
        doStoreSample = true;
    else
        % 'auto'
        if totalElements <= opts.maxStoreElements
            doStoreAll = true;
        else
            doStoreSample = true; % default fallback: sample
        end
    end

    % ----------------------------
    % Report struct (always keep metrics; vectors are optional)
    % ----------------------------
    Report = struct();
    Report.MSE = MSE;
    Report.RMSE = RMSE;
    Report.MAE = MAE;
    Report.SSE = SSE;
    Report.SST = SST;
    Report.Rsquared = Rsq;
    Report.Bias = Bias;

    Report.NumOfOutput = nOut;
    Report.NumSamples  = N;

    if nOut == 1
        Report.MeanAbsoluteError = MAE(1);
    end

    % Vector storage policy
    Report.StorePolicy = struct();
    Report.StorePolicy.mode = storeMode;
    Report.StorePolicy.totalElements = totalElements;
    Report.StorePolicy.maxStoreElements = opts.maxStoreElements;

    if doStoreNone
        % Do not store vectors
        Report.Prediction = [];
        Report.ErrorVector = [];
        Report.StorePolicy.stored = 'none';
    elseif doStoreAll
        Report.Prediction = prediction;
        Report.ErrorVector = err;
        Report.StorePolicy.stored = 'all';
    else
        % sample
        rng(opts.randomSeed);
        nKeep = min(opts.sampleStoreN, N);
        idx = randperm(N, nKeep);

        Report.Prediction = prediction(:, idx);
        Report.ErrorVector = err(:, idx);
        Report.StorePolicy.stored = 'sample';
        Report.StorePolicy.sampleN = nKeep;
    end

    % ----------------------------
    % Print performance summary
    % ----------------------------
    if opts.print
        printSummary(Rsq, MAE, RMSE, Bias, MSE, nOut, N, Report.StorePolicy);
    end

    % ----------------------------
    % Visualization (with downsampling safeguard)
    % ----------------------------
    if opts.plot
        plotReport(label, prediction, err, opts);
    end
end

% =========================================================
% Printing
% =========================================================
function printSummary(Rsq, MAE, RMSE, Bias, MSE, nOut, N, storePolicy)
    fprintf('--- FittingReport ---\n');
    fprintf('Samples: %d, Outputs: %d\n', N, nOut);

    if isstruct(storePolicy) && isfield(storePolicy,'stored')
        fprintf('Vector storage: %s', storePolicy.stored);
        if strcmp(storePolicy.stored,'sample') && isfield(storePolicy,'sampleN')
            fprintf(' (N=%d)', storePolicy.sampleN);
        end
        fprintf('\n');
    end

    fprintf('  %-6s %-12s %-12s %-12s %-12s %-12s\n', ...
        'y', 'R^2', 'MAE', 'RMSE', 'Bias', 'MSE');

    for i = 1:nOut
        fprintf('  %-6s %-12.6g %-12.6g %-12.6g %-12.6g %-12.6g\n', ...
            ['y' num2str(i)], Rsq(i), MAE(i), RMSE(i), Bias(i), MSE(i));
    end
    fprintf('---------------------\n\n');
end

% =========================================================
% Plotting (downsample large N)
% =========================================================
function plotReport(label, prediction, err, opts)
    nOut = size(label,1);
    N    = size(label,2);

    % Downsample indices for scatter to keep plotting responsive
    if N > opts.maxPlotPoints
        rng(opts.randomSeed);
        idxPlot = randperm(N, opts.maxPlotPoints);
    else
        idxPlot = 1:N;
    end

    % --- Predict vs Actual ---
    figure('Name','Predict vs Actual');
    for i = 1:nOut
        subplot(1,nOut,i);
        scatter(prediction(i,idxPlot), label(i,idxPlot), ...
            opts.markerSize, 'k', 'filled');
        hold on; grid on;

        minV = min([label(i,idxPlot), prediction(i,idxPlot)]);
        maxV = max([label(i,idxPlot), prediction(i,idxPlot)]);
        plot([minV maxV],[minV maxV],'LineWidth',2);

        xlabel('Prediction');
        ylabel('Actual');
        title(['Predict vs Actual (y' num2str(i) ')']);
        axis tight;
    end

    % --- Error Distribution ---
    figure('Name','Error Distribution');
    for i = 1:nOut
        subplot(1,nOut,i);

        % Histogram also can be heavy; sample if needed
        ePlot = err(i,idxPlot);

        if exist('histogram','file') == 2
            histogram(ePlot, opts.numHistBins);
        else
            hist(ePlot, opts.numHistBins);
        end

        hold on; grid on;
        yl = ylim;
        plot([0 0], yl, 'LineWidth',2);
        ylim(yl);

        xlabel('Error');
        ylabel('Count');
        title(['Error Distribution (y' num2str(i) ')']);
    end

    if N > numel(idxPlot)
        fprintf('[FittingReport] Plot downsample: %d -> %d points\n', N, numel(idxPlot));
    end
end

% =========================================================
% Utility
% =========================================================
function opts = applyDefaults(opts, defaults)
    f = fieldnames(defaults);
    for k = 1:numel(f)
        name = f{k};
        if ~isfield(opts,name) || isempty(opts.(name))
            opts.(name) = defaults.(name);
        end
    end
end