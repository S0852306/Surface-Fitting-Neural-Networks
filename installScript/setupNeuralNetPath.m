function info = setupNeuralNetPath(opts)
% setupNeuralNetPath
%
% Add NeuralNetsPack folders to MATLAB path:
%   - <packRoot>/functions (with subfolders)
%   - <packRoot>/demoScript
%   - <packRoot>/installScript
%
% Usage from package root:
%   run('installScript/installNeuralNetsPack.m')
%
% Usage after installScript is on path:
%   setupNeuralNetPath();
%   setupNeuralNetPath(struct('savePath',false));
%
% Options:
%   retireOldVersions (default: true)
%   savePath          (default: true)
%   verbose           (default: true)
%   packageTokens     (default: common NeuralNetsPack folder names)

    if nargin < 1 || isempty(opts)
        opts = struct();
    end

    defaults = struct( ...
        'retireOldVersions', true, ...
        'savePath', true, ...
        'verbose', true, ...
        'packageTokens', {{ ...
            'NeuralNetsPack', ...
            'Surface-Fitting-Neural-Networks' ...
        }} ...
    );
    opts = applyDefaults(opts, defaults);

    thisFile = mfilename('fullpath');
    installRoot = fileparts(thisFile);
    packRoot = fileparts(installRoot);
    [~, packFolderName] = fileparts(packRoot);

    functionRoot = fullfile(packRoot, 'functions');
    demoScriptRoot = fullfile(packRoot, 'demoScript');

    if ~isfolder(functionRoot)
        error('setupNeuralNetPath:MissingFunctionsFolder', ...
            '"functions" folder not found under:\n  %s', packRoot);
    end

    removed = {};
    if opts.retireOldVersions
        removed = retirePackFromPath(packRoot, packFolderName, opts.packageTokens);
    end

    added = {};
    addpath(installRoot);
    added{end+1} = installRoot;

    addpath(genpath(functionRoot));
    added{end+1} = functionRoot;

    if isfolder(demoScriptRoot)
        addpath(demoScriptRoot);
        added{end+1} = demoScriptRoot;
    end

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

    info = struct();
    info.packRoot = packRoot;
    info.installRoot = installRoot;
    info.functionRoot = functionRoot;
    info.demoScriptRoot = demoScriptRoot;
    info.addedEntries = added;
    info.removedEntries = removed;
    info.savedPath = didSave;

    if opts.verbose
        fprintf('[%s] Enabled paths:\n', packFolderName);
        for idx = 1:numel(added)
            fprintf('  %s\n', added{idx});
        end
        fprintf('[%s] Retired %d old path entr%s.\n', ...
            packFolderName, numel(removed), pluralY(numel(removed)));
        fprintf('[%s] savepath: %s\n', ...
            packFolderName, ternary(didSave, 'OK', 'FAILED'));
    end
end

function opts = applyDefaults(opts, defaults)
    names = fieldnames(defaults);
    for idx = 1:numel(names)
        name = names{idx};
        if ~isfield(opts, name) || isempty(opts.(name))
            opts.(name) = defaults.(name);
        end
    end
end

function removed = retirePackFromPath(currentPackRoot, packFolderName, packageTokens)
    removed = {};
    entries = strsplit(path, pathsep);
    packageTokens = unique([{packFolderName}, packageTokens]);
    currentPackRoot = normalizePath(currentPackRoot);

    for idx = 1:numel(entries)
        entry = entries{idx};
        if isempty(entry)
            continue;
        end

        normalizedEntry = normalizePath(entry);
        if isCurrentPackPath(normalizedEntry, currentPackRoot)
            continue;
        end

        if isKnownPackPath(normalizedEntry, packageTokens)
            rmpath(entry);
            removed{end+1} = entry; %#ok<AGROW>
        end
    end
end

function normalized = normalizePath(pathText)
    normalized = char(pathText);
    normalized = strrep(normalized, '/', filesep);
    normalized = regexprep(normalized, [regexptranslate('escape', filesep) '+'], filesep);
    if ispc
        normalized = lower(normalized);
    end
end

function tf = isCurrentPackPath(pathEntry, currentPackRoot)
    tf = strcmp(pathEntry, currentPackRoot) || startsWith(pathEntry, [currentPackRoot filesep]);
end

function tf = isKnownPackPath(pathEntry, packageTokens)
    tf = false;
    for idx = 1:numel(packageTokens)
        token = normalizePath(packageTokens{idx});
        if contains(pathEntry, [filesep token filesep]) || endsWith(pathEntry, [filesep token])
            tf = true;
            return
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

function out = ternary(condition, trueValue, falseValue)
    if condition
        out = trueValue;
    else
        out = falseValue;
    end
end
