function [B,dB] = BSplineBasis(z, grid)
%BSPLINEBASIS  Evaluate B-spline basis and optionally derivative basis.
%   [B, dB] = BSplineBasis(z, grid)
%
%   z    : pre-activation values (any shape, flattened internally)
%   grid : struct with precomputed .P, .dP, .invH, .numGrid, etc.
%
%   B  : [numel(z) x numBasis]  basis values
%   dB : [numel(z) x numBasis]  derivative of basis w.r.t. z (optional)
%
%   Uses precomputed polynomial coefficients (Horner evaluation).
%   Complex z is supported for complex-step differentiation:
%   real(z) is used for span lookup; full z for local coordinate
%   so the imaginary perturbation propagates correctly.

    k        = grid.order;
    numBasis = grid.numBasis;
    numG     = grid.numGrid;
    gMin     = grid.gridRange(1);
    invH     = grid.invH;
    Ppre     = grid.P;               % [numGrid x k x k]

    zr = real(z(:));                 % real part for span lookup
    zv = z(:);                       % full (possibly complex)
    N  = numel(zv);

    % --- Span lookup ---
    zClamped = max(gMin, min(grid.gridRange(2), zr));
    s = floor((zClamped - gMin) * invH);       % 0-indexed span
    s = min(s, numG - 1);

    % --- Local coordinate u ∈ [0,1) (full z for complex-step) ---
    u = (zv - gMin) * invH - s;                % [N x 1]

    % --- Gather precomputed poly coefficients for each point's span ---
    s1 = s + 1;                                 % 1-indexed span
    Pn = reshape(Ppre(s1, :, :), N, k, k);     % [N x k x k]

    % --- Horner evaluation of k basis polynomials ---
    bVals = Pn(:, :, 1);                        % leading coefficients
    for p = 2:k
        bVals = bVals .* u + Pn(:, :, p);
    end

    % --- Scatter into [N x numBasis] sparse matrix ---
    cIdx = s + (1:k);                            % [N x k], 1-indexed column
    rows = repmat((1:N)', 1, k);
    B = full(sparse(rows(:), cIdx(:), bVals(:), N, numBasis));

    % --- Derivative basis: dB/dz ---
    if nargout >= 2
        dPpre = grid.dP;                        % [numGrid x k x (k-1)]
        dPn = reshape(dPpre(s1, :, :), N, k, k - 1);

        dbVals = dPn(:, :, 1);
        for p = 2:k-1
            dbVals = dbVals .* u + dPn(:, :, p);
        end

        dB = full(sparse(rows(:), cIdx(:), dbVals(:), N, numBasis));
    end
end
