function dc = BSplineCoeffGrad(dLda, z, grid)
%BSPLINECOEFFGRAD  Gradient of loss w.r.t. B-spline coefficients.
%   dc = BSplineCoeffGrad(dLda, z, grid)
%
%   dLda : upstream gradient dL/da  (same size as z)
%   z    : pre-activation values    (same size as z)
%   grid : B-spline grid struct
%
%   Returns dc : [numBasis x 1] gradient vector.

    B  = BSplineBasis(z, grid);        % [N x numBasis]
    dc = B' * dLda(:);                 % [numBasis x 1]
end
