function [a, dadz] = BSplineActivation(z, c, grid)
%BSPLINEACTIVATION  B-spline based learnable activation.
%   [a, dadz] = BSplineActivation(z, c, grid)
%
%   z    : pre-activation  (any size matrix)
%   c    : [numBasis x 1]  learnable B-spline coefficients
%   grid : struct from Initialization (contains .knots, .order, .numBasis)
%
%   Returns:
%     a    : activation values  (same size as z)
%     dadz : derivative da/dz   (same size as z)

    sz = size(z);

    if nargout >= 2
        [B, dB] = BSplineBasis(z, grid);
        dadz = reshape(dB * c, sz);
    else
        B = BSplineBasis(z, grid);
    end

    a = reshape(B * c, sz);
end
