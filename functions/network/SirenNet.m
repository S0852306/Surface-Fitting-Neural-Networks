function FunctionOutput = SirenNet(data, NN)
% ANN_SirenNet forward pass (SIREN)
% IO unchanged: FunctionOutput = ANN_SirenNet(data, NN)
%
% Required NN fields (suggested):
%   NN.depth
%   NN.weight{1..depth}, NN.bias{1..depth}
%   NN.InputScaleVector, NN.InputCenterVector
%   NN.OutActive (e.g., @(x)x)
%
% Optional SIREN params:
%   NN.siren.omega0      (default 30)
%   NN.siren.omegaHidden (default = omega0)
%
% Hidden activation:
%   v = sin(omega * z)

    % ------------------------------------------------------------
    % 0) Input scaling (same as your ANN)
    % ------------------------------------------------------------
    v = NN.InputScaleVector .* data - NN.InputCenterVector;

    % ------------------------------------------------------------
    % 0.5) SIREN params (safe defaults)
    % ------------------------------------------------------------
    omega0 = 30;
    omegaHidden = [];

    if isfield(NN, 'siren')
        if isfield(NN.siren, 'omega0') && ~isempty(NN.siren.omega0)
            omega0 = NN.siren.omega0;
        end
        if isfield(NN.siren, 'omegaHidden') && ~isempty(NN.siren.omegaHidden)
            omegaHidden = NN.siren.omegaHidden;
        end
    end
    if isempty(omegaHidden)
        omegaHidden = omega0;
    end

    % ------------------------------------------------------------
    % 1) Hidden layers (SIREN)
    % ------------------------------------------------------------
    for i = 1:(NN.depth - 1)
        z = NN.weight{i} * v + NN.bias{i};  % pre-activation

        if i == 1
            v = sin(omega0 * z);
        else
            v = sin(omegaHidden * z);
        end
    end

    % ------------------------------------------------------------
    % 2) Output layer (same as your ANN)
    % ------------------------------------------------------------
    zOut = NN.weight{NN.depth} * v + NN.bias{NN.depth};
    FunctionOutput = NN.OutActive(zOut);

end