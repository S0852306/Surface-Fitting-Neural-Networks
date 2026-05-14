function [a, dadz] = SplineActivation(z, c, grid)
%SPLINEACTIVATION  Differentiable linear-spline activation.
%   [a, dadz] = SplineActivation(z, c, grid)
%
%   z     : pre-activation  (any size matrix)
%   c     : [numKnots x 1]  learnable spline coefficients
%   grid  : struct with .knots (1 x numKnots), .h (scalar), .numIntervals
%
%   Linear interpolation inside [gridMin, gridMax].
%   Linear extrapolation outside (using slope of first/last interval).

    knots = grid.knots;        % 1 x numKnots
    h     = grid.h;
    G     = grid.numIntervals; % number of intervals

    % --- interval index (1-based, clamped to [1, G]) ---
    %     Use real(z) so complex-step perturbations don't break indexing.
    zr  = real(z);
    idx = floor((zr - knots(1)) / h) + 1;
    idx = max(1, min(G, idx));

    % --- gather knot values ---
    c_left  = c(idx);          % c_{k}
    c_right = c(idx + 1);     % c_{k+1}
    t_left  = knots(idx);     % t_{k}

    % --- linear interp / extrap (z may be complex; imaginary part propagates) ---
    frac = (z - t_left) / h;
    a = c_left .* (1 - frac) + c_right .* frac;

    % --- derivative da/dz ---
    if nargout >= 2
        dadz = (c_right - c_left) / h;
    end
end
