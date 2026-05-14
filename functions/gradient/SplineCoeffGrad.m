function dc = SplineCoeffGrad(dLda, z, grid)
%SPLINECOEFFGRAD  Gradient of loss w.r.t. spline coefficients.
%   dc = SplineCoeffGrad(dLda, z, grid)
%
%   dLda : upstream gradient dL/da  (same size as z)
%   z    : pre-activation values    (same size as z)
%   grid : struct with .knots, .h, .numIntervals, .numKnots
%
%   Returns dc : [numKnots x 1] gradient vector.

    knots    = grid.knots;
    h        = grid.h;
    G        = grid.numIntervals;
    numKnots = grid.numKnots;

    % --- interval index (clamped, same as forward) ---
    idx = floor((z - knots(1)) / h) + 1;
    idx = max(1, min(G, idx));

    t_left  = knots(idx);
    left_w  = 1 - (z - t_left) / h;   % da/dc_k
    right_w = (z - t_left) / h;        % da/dc_{k+1}

    % --- accumulate into coefficient gradient ---
    dc = accumarray(idx(:),     dLda(:) .* left_w(:),  [numKnots, 1]) + ...
         accumarray(idx(:) + 1, dLda(:) .* right_w(:), [numKnots, 1]);
end
