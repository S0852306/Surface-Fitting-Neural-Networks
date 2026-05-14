%% Install NeuralNetsPack
% Run this file once after downloading the package.
% It saves the MATLAB path so NeuralFit works from any current folder.

thisFile = mfilename('fullpath');
installRoot = fileparts(thisFile);
addpath(installRoot);

info = setupNeuralNetPath(struct( ...
    'retireOldVersions', true, ...
    'savePath', true, ...
    'verbose', true ...
));

fprintf('\nInstallation complete.\n');
fprintf('You can now call NeuralFit from any MATLAB current folder.\n');
