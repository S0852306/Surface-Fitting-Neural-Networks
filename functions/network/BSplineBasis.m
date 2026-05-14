function [B, dB] = BSplineBasis(z, grid)
%BSPLINEBASIS  Evaluate B-spline basis and optionally derivative basis.
%   [B, dB] = BSplineBasis(z, grid)
%
%   z    : pre-activation values (any shape, flattened internally)
%   grid : struct with .knots (augmented), .order, .numBasis, .gridRange
%
%   B  : [numel(z) x numBasis]  basis values
%   dB : [numel(z) x numBasis]  derivative of basis w.r.t. z (optional)
%
%   Uses Cox-de Boor recursion.  Complex z is supported for complex-step
%   differentiation: real(z) is used for span indicators; full z for
%   arithmetic so the imaginary perturbation propagates correctly.
%   Points outside gridRange get polynomial extrapolation (degree k-1).

    knots    = grid.knots;       % augmented knot vector (1 x numKnots)
    k        = grid.order;       % B-spline order (degree = k-1)
    numBasis = grid.numBasis;
    gMin     = grid.gridRange(1);
    gMax     = grid.gridRange(2);

    zr = real(z(:));             % real part for indicator tests
    zv = z(:);                   % full (possibly complex) for blending
    N  = numel(zv);

    numKnots = numel(knots);
    nSpans   = numKnots - 1;

    % Clamp real part to grid range (extrapolation support)
    zInd = max(gMin, min(gMax, zr));

    % --- Order 1: piecewise constant indicators ---
    Bc = zeros(N, nSpans);
    for i = 1:nSpans
        if knots(i+1) > knots(i)   % skip zero-width spans
            Bc(:,i) = double(zInd >= knots(i) & zInd < knots(i+1));
        end
    end
    % Include right endpoint in last non-zero-width span
    lastNZ = find(diff(knots) > 0, 1, 'last');
    Bc(zInd == gMax, lastNZ) = 1;

    % --- Cox-de Boor recursion (order 2 .. k) ---
    BprevForDeriv = [];
    for p = 2:k
        if p == k
            BprevForDeriv = Bc;   % save order k-1 for derivative
        end
        nB   = nSpans - p + 1;
        Bnew = zeros(N, nB);
        for i = 1:nB
            d1 = knots(i+p-1) - knots(i);
            d2 = knots(i+p)   - knots(i+1);
            w1 = 0;  w2 = 0;
            if d1 > 0
                w1 = (zv - knots(i))   / d1 .* Bc(:,i);
            end
            if d2 > 0
                w2 = (knots(i+p) - zv) / d2 .* Bc(:,i+1);
            end
            Bnew(:,i) = w1 + w2;
        end
        Bc = Bnew;
    end

    B = Bc;  % [N x numBasis]

    % --- Derivative basis: dB_{i,k}/dz ---
    if nargout >= 2
        dB  = zeros(N, numBasis);
        km1 = k - 1;
        for i = 1:numBasis
            d1 = knots(i+k-1) - knots(i);
            d2 = knots(i+k)   - knots(i+1);
            t1 = 0;  t2 = 0;
            if d1 > 0
                t1 = km1 / d1 * BprevForDeriv(:,i);
            end
            if d2 > 0
                t2 = km1 / d2 * BprevForDeriv(:,i+1);
            end
            dB(:,i) = t1 - t2;
        end
    end
end
