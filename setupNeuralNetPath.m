function info = setupNeuralNetPath(opts)
% setupNeuralNetPath
%
% Add NeuralNetPack folders to MATLAB path:
%   - <packRoot>/functions (with subfolders)
%   - <packRoot>/+demoScript
%   - <packRoot>/+liveDemo
%
% Usage:
%   setupNeuralNetPath();
%   setupNeuralNetPath(struct('savePath',false));
%
% Options:
%   retireOldVersions (default: true)
%   savePath          (default: true)
%   verbose           (default: true)

    if nargin < 1 || isempty(opts)
        opts = struct();
    end

    % -----------------------------
    % Defaults
    % -----------------------------
    defaults = struct( ...
        'retireOldVersions', true, ...
        'savePath', true, ...
        'verbose', true ...
    );

    opts = applyDefaults(opts, defaults);

    % -----------------------------
    % Detect pack root
    % -----------------------------
    thisFile = mfilename('fullpath');
    packRoot = fileparts(thisFile);
    [~, packFolderName] = fileparts(packRoot);

    functionRoot   = fullfile(packRoot, 'functions');
    demoScriptRoot = fullfile(packRoot, 'demoScript');
    liveDemoRoot   = fullfile(packRoot, 'liveDemo');

    if ~isfolder(functionRoot)
        error('setupNeuralNetPath:MissingFunctionsFolder', ...
            '"functions" folder not found under:\n  %s', packRoot);
    end

    % -----------------------------
    % Retire old versions
    % -----------------------------
    removed = {};
    if opts.retireOldVersions
        removed = retirePackFromPath(packRoot, packFolderName);
    end

    % -----------------------------
    % Add folders
    % -----------------------------
    added = {};

    % functions (include subfolders)
    addpath(genpath(functionRoot));
    added{end+1} = functionRoot;

    % +demoScript
    if isfolder(demoScriptRoot)
        addpath(demoScriptRoot);
        added{end+1} = demoScriptRoot;
    end

    % +liveDemo
    if isfolder(liveDemoRoot)
        addpath(liveDemoRoot);
        added{end+1} = liveDemoRoot;
    end

    % -----------------------------
    % Save path if requested
    % -----------------------------
    didSave = false;
    if opts.savePath
        try
            savepath;
            didSave = true;
        catch me
            warning('setupNeuralNetPath:SavePathFailed', ...
                'savepath failed: %s', me.message);
        end
    end

    % -----------------------------
    % Output info
    % -----------------------------
    info = struct();
    info.packRoot      = packRoot;
    info.functionRoot  = functionRoot;
    info.demoScriptRoot= demoScriptRoot;
    info.liveDemoRoot  = liveDemoRoot;
    info.addedEntries  = added;
    info.removedEntries= removed;
    info.savedPath     = didSave;

    if opts.verbose
        fprintf('[%s] Enabled paths:\n', packFolderName);
        for i = 1:numel(added)
            fprintf('  %s\n', added{i});
        end
        fprintf('[%s] Retired %d old path entr%s.\n', ...
            packFolderName, numel(removed), pluralY(numel(removed)));
        fprintf('[%s] savepath: %s\n', ...
            packFolderName, ternary(didSave,'OK','FAILED'));
    end
end

% =========================================================
% Helper Functions (local)
% =========================================================

function opts = applyDefaults(opts, defaults)
    f = fieldnames(defaults);
    for i = 1:numel(f)
        k = f{i};
        if ~isfield(opts, k) || isempty(opts.(k))
            opts.(k) = defaults.(k);
        end
    end
end

function removed = retirePackFromPath(currentPackRoot, packFolderName)
% Remove all path entries belonging to this pack

    removed = {};
    p = path;
    parts = strsplit(p, pathsep);

    packToken1 = [filesep packFolderName filesep];
    packToken2 = [filesep packFolderName];

    for i = 1:numel(parts)
        entry = parts{i};
        if isempty(entry)
            continue;
        end

        isPackPath = contains(entry, packToken1) || endsWith(entry, packToken2);

        if ~isPackPath
            continue;
        end

        % Don't remove current root
        if contains(entry, currentPackRoot)
            continue;
        end

        if exist(entry, 'dir') == 7
            rmpath(entry);
            removed{end+1} = entry; %#ok<AGROW>
        end
    end
end

function s = pluralY(n)
    if n == 1
        s = 'y';
    else
        s = 'ies';
    end
end

function out = ternary(cond, a, b)
    if cond
        out = a;
    else
        out = b;
    end
end