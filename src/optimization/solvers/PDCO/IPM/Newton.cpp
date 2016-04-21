/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>
#include "./util.hpp"

namespace El {
namespace pdco {

// The following solves the following convex problem:
//
//   minimize phi(x) + 1/2 ||D1*x||^2 + 1/2 ||r||^2
//     x,r
//   s.t.     A*x + D2*r = b, bl <= x <= bu, r unconstrained
//
// We do so by applying Newton's method to the problem
//
//   minimize phi(x) + 1/2 ||D1*x||^2 + 1/2 ||r||^2 
//                   + mu_k*sum{log(x-bl) + log(bu - x)}
//     x,r
//   s.t.     A*x + D2*r = b, r unconstrained
// with mu_k -> 0

template<typename Real>
void Newton
( const PDCOObj<Real>& phi,
  const Matrix<Real>& A,
  const Matrix<Real>& b, 
  const Matrix<Real>& bl,
  const Matrix<Real>& bu,
  const Matrix<Real>& D1,
  const Matrix<Real>& D2,
        Matrix<Real>& x,
        Matrix<Real>& r,
        Matrix<Real>& y,
        Matrix<Real>& z,
  const PDCOCtrl<Real>& ctrl )
{
    DEBUG_ONLY(CSE cse("pdco::Newton")) 

    Output("=========== Beginning Newton's Method ===========");

    // Useful defined variables
    Int m = A.Height();
    Int n = A.Width();
    vector<Int> ALL_m = IndexRange(m);
    vector<Int> ALL_n = IndexRange(n);
    vector<Int> ZERO (1,0);
    Matrix<Real> bCopy; // Needed for fixed variables
    Matrix<Real> D2sq;  // D2sq = D2^2
    Matrix<Real> xFix;  // Index set for fixed variables
    Matrix<Real> xLow;  // Index set for lower bounded variables
    Matrix<Real> xUpp;  // Index set for upper bounded variables
    Matrix<Real> grad;  // For gradient of phi
    Matrix<Real> Hess;  // For hessian of phi
    Matrix<Real> H;     // Used for KKT system
    Matrix<Real> ACopy; // Used for KKT system
    Matrix<Real> At;    // A^T (after fixed variables removed)
    Matrix<Real> AtCopy;// Copy of A^T
    Matrix<Real> w;     // Residual for KKT system
    Matrix<Real> dx;    // Primal step direction
    Matrix<Real> dy;    // Dual step direction
    Matrix<Real> dz1;   // Complementarity step direction
    Matrix<Real> dz2;   // Complementarity step direction
    Real center;        // Centering parameter
    Real Pfeas;         // Primal feasibility
    Real Dfeas;         // Dual feasibility
    Real Cfeas;         // Complementarity feasibility
    Real Cinf0;         // Complementarity convergence criteria
    Real stepx;
    Real stepz;
    Real stepmu;
    Real mulast = 0.1*ctrl.optTol;

    Copy(D2, D2sq);
    DiagonalScale(LEFT, NORMAL, D2, D2sq);

    // Determine index sets for lower bounded variables,
    // upper bounded variables, and fixed variables
    vector<Int> ixSetLow;
    vector<Int> ixSetUpp;
    vector<Int> ixSetFix;
    pdco::ClassifyBounds(bl, bu, ixSetLow, ixSetUpp, ixSetFix, ctrl.print);

    if( ixSetFix.size() > 0 )
    {
        // Fix b to allow for fixed variables
        Matrix<Real> Asub;
        GetSubmatrix(bl, ixSetFix, ZERO, xFix); // xFix = bl(ixSetFix)
        GetSubmatrix(A, ALL_m, ixSetFix, Asub); // Asub = A(:,ixSetFix)
        Copy(b, bCopy);
        Gemv(NORMAL, Real(-1), Asub, xFix, Real(1), bCopy); // b = b - A*xFix
    }
    else
    {
        Copy(b, bCopy);
    }

    // Scale input data?

    // Initialize the data
    Matrix<Real> z1;
    Matrix<Real> z2;
    pdco::Initialize(x, y, z1, z2, bl, bu, 
      ixSetLow, ixSetUpp, ixSetFix,ctrl.x0min, ctrl.z0min, m, n, ctrl.print);

    // Compute residuals
    // Residual vectors to be populated
    Matrix<Real> r1;
    Matrix<Real> r2;

    phi.grad(x, grad); // get gradient
    ResidualPD(A, ixSetLow, ixSetUpp, ixSetFix,
      bCopy, D1, D2, grad, x, y, z1, z2, r1, r2);

    Matrix<Real> cL;
    Matrix<Real> cU;
    Real mu = Max(ctrl.mu0,mulast);
    ResidualC(mu, ixSetLow, ixSetUpp, bl, bu, x, z1, z2, center, Cinf0, cL, cU);

    // Get Hessian
    phi.hess(x, Hess);

    Pfeas = InfinityNorm(r1);
    Dfeas = InfinityNorm(r2);
    Cfeas = Max(InfinityNorm(cL), InfinityNorm(cU));

    Output("  Pfeas = ", Pfeas);
    Output("  Dfeas = ", Dfeas);
    Output("  Cinf0 = ", Cinf0);

    Matrix<Real> zeros;

    Copy(A, ACopy); // ACopy = A'
    // Make copy of A with zero columns for fixed variables
    if( ixSetFix.size() > 0 )
    {
        Zeros(zeros, ixSetFix.size(), n);
        SetSubmatrix(ACopy, ALL_m, ixSetFix, zeros);
    }
    // Get transpose
    Transpose(ACopy, At);

    // Main loop
    for( Int numIts=0; numIts<=ctrl.maxIts; ++numIts )
    {
        Output("========== Iteration: ", numIts, " ==========");
        Output("  mu = ", mu);
        switch( ctrl.method )
        {
            case Method::LDLy:
            {
                FormHandW( Hess, D1, x, z1, z2, bl, bu,
                  ixSetLow, ixSetUpp, ixSetFix, r2, cL, cU, H, w );

                if( ixSetFix.size() > 0 )
                {
                    // Set rows/cols corresponding to fixed variables to zero
                    Zeros(zeros, ixSetFix.size(), n);
                    Matrix<Real> ones;
                    Ones(ones, ixSetFix.size(), 1);
                    SetSubmatrix(H, ixSetFix, ALL_n, zeros);
                    Zeros(zeros, n, ixSetFix.size());
                    SetSubmatrix(H, ALL_n, ixSetFix, zeros);
                    // Fix diagonal to 1
                    SetSubmatrix(H, ixSetFix, ixSetFix, ones);
                }

//                Print(H, "H");
//                Print(w, "w");
//                Print(r1, "r1");
//                Print(r2, "r2");
//                Print(cL, "cL");
//                Print(cU, "cU");

                // Make a copy of At
                Copy(At, AtCopy);

                // Start solving for dx, dy
                // TODO: Use pivoted?
                Matrix<Real> S; // S = A*(H\A') + D2^2
                Zeros(S, m, m);
                LDL(H, false);
                ldl::SolveAfter(H, AtCopy, false); // ACopy = H\A'
                Gemm(NORMAL, NORMAL, Real(1), ACopy, AtCopy, Real(0), S); // S = A*(H\A')
                UpdateDiagonal(S, Real(1), D2sq, 0); // S = A*(H\A) + D2^2
                Copy(r1, dy); // dy = r1
                ldl::SolveAfter(H, w, false); // w = H\w
                Gemv(NORMAL, Real(-1), ACopy, w, Real(1), dy); // dy = r1 - A*(H\w)

                LDL(S, false);
                ldl::SolveAfter(S, dy, false); // dy = S\(r1 - A*(H\w))

                // Compute dx
                Copy(w, dx); // dx = H\w
                Gemv(NORMAL, Real(1), AtCopy, dy, Real(1), dx); // dx = H\w + (H\A')*dy

//                Print(dx, "dx");
//                Print(dy, "dy");

                // Compute dz1
                Matrix<Real> tmp1;
                Matrix<Real> tmp2;
                Copy(x, tmp1);
                tmp1 -= bl;
                GetSubmatrix(tmp1, ixSetLow, ZERO, tmp2);
                GetSubmatrix(dx, ixSetLow, ZERO, dz1);
                DiagonalScale(LEFT, NORMAL, z1, dz1);
                dz1 *= -1;
                dz1 += cL;
                DiagonalSolve(LEFT, NORMAL, tmp2, dz1); // dz1 = (x-bl)^-1 * (cL - z1*dx)

                // Compute dz2
                Copy(bu, tmp1);
                tmp1 -= x;
                GetSubmatrix(tmp1, ixSetUpp, ZERO, tmp2);
                GetSubmatrix(dx, ixSetUpp, ZERO, dz2);
                DiagonalScale(LEFT, NORMAL, z2, dz2);
                dz2 += cU;
                DiagonalSolve(LEFT, NORMAL, tmp2, dz2); // dz1 = (x-bl)^-1 * (cL - z1*dx)

//                Print(dz1, "dz1");
//                Print(dz2, "dz2");
            }
                break;
            case Method::LDLx:
                RuntimeError("LDLx not yet implemented.");
                break;
            case Method::LDL2:
                RuntimeError("LDLy not yet implemented.");
                break;
            default:
                RuntimeError("Unrecognized method option.");
        }

        // dx, dy, dz1, dz2 should be computed at this point
        // Return stepx, stepz
        bool success = Linesearch(phi, mu, ACopy, bCopy, bl, bu, D1, D2, 
          x, y, z1, z2, r1, r2, center, Cinf0, cL, cU, stepx, stepz,
          dx, dy, dz1, dz2, ixSetLow, ixSetUpp, ixSetFix, ctrl);

        if( !success )
        {
            // Linesearch failed...what now?
            Output("Linesearch failed at iteration: ", numIts);
        }

        // Check convergence criteria
        Pfeas = InfinityNorm(r1);
        Dfeas = InfinityNorm(r2);
        Cfeas = Max(InfinityNorm(cL), InfinityNorm(cU));
        bool converged = (Pfeas <= ctrl.feaTol)
          && (Dfeas <= ctrl.feaTol)
          && (Cinf0 <= ctrl.optTol);

        Output("  Pfeas    = ", Pfeas);
        Output("  Dfeas    = ", Dfeas);
        Output("  Cinf0    = ", Cinf0);
        Output("  ||cL||oo = ", InfinityNorm(cL));
        Output("  ||cU||oo = ", InfinityNorm(cU));
        Output("  center   = ", center);

        if (true)
        {
            // Update mu
            // TODO: Clarify this
            stepmu = Min(stepx, stepz);
            stepmu = Min(stepmu, ctrl.stepTol);
            Real muold = mu;
            Real mumin = Max(Pfeas, Dfeas);
            mumin = 0.1*Max(mumin, Cfeas);
            mumin = Min(mu, mumin);
            mu = mu - stepmu*mu;

            if( center >= ctrl.bigcenter )
            {
                mu = muold;
            }
            mu = Max(mu, mumin);
            mu = Max(mu, mulast);
        }
        else
        {
            if( (Pfeas <= ctrl.feaTol)
               && (Dfeas <= ctrl.feaTol) )
            {
                mu *= 0.5;
            }
        }

        // Update gradient and Hessian
        phi.grad(x, grad);
        phi.hess(x, Hess);

        // Recompute residuals
        ResidualPD(A, ixSetLow, ixSetUpp, ixSetFix,
          bCopy, D1, D2, grad, x, y, z1, z2, r1, r2);

        ResidualC(mu, ixSetLow, ixSetUpp, bl, bu, x, z1, z2, center, Cinf0, cL, cU);

        if (converged)
            break;
    }

    // Reconstruct solution
    // scale?
    // set x(fix) = bl(fix);
    Matrix<Real> tmp;
    GetSubmatrix(bl, ixSetFix, ZERO, tmp);
    SetSubmatrix(x, ixSetFix, ZERO, tmp); 

    // Report active constraints
    Int lowerActive = 0;
    Int upperActive = 0;
    Int ctrLow = 0;
    Int ctrUpp = 0;
    for( Int i = 0; i < n; i++ )
    {
        if( i == ixSetLow[ctrLow])
        {
            if( x.Get(i,0) - bl.Get(i,0) < 1e-8 )
                lowerActive++;
            ctrLow++;
        }
        if( i == ixSetUpp[ctrUpp])
        {
            if( bu.Get(i,0) - x.Get(i,0) < 1e-8 )
                upperActive++;
            ctrUpp++;
        }
    }

    // Reconstruct z from z1 and z2
    Zeros(z, n, 1);
    UpdateSubmatrix(z, ixSetLow, ZERO, Real(1), z1);
    UpdateSubmatrix(z, ixSetUpp, ZERO, Real(-1), z2);
    Matrix<Real> zFix;
    GetSubmatrix(grad, ixSetFix, ZERO, zFix);
    GetSubmatrix(r2, ixSetFix, ZERO, tmp);
    Axpy(Real(-1), tmp, zFix);
    UpdateSubmatrix(z, ixSetFix, ZERO, Real(1), zFix);

    Output("=========== Completed Newton's Method ===========");

    Output("  Pfeas    = ", Pfeas);
    Output("  Dfeas    = ", Dfeas);
    Output("  Cinf0    = ", Cinf0);
    Output("  ||cL||oo = ", InfinityNorm(cL));
    Output("  ||cU||oo = ", InfinityNorm(cU));
    Output("  center   = ", center);

    Output("  Number active constraints on");
    Output("    Lower bound: ", lowerActive);
    Output("    Upper bound: ", upperActive);
}

#define PROTO(Real) \
  template void Newton \
  ( const pdco::PDCOObj<Real>& phi, \
    const Matrix<Real>& A, \
    const Matrix<Real>& b, \
    const Matrix<Real>& bl, \
    const Matrix<Real>& bu, \
    const Matrix<Real>& D1, \
    const Matrix<Real>& D2, \
          Matrix<Real>& x, \
          Matrix<Real>& r, \
          Matrix<Real>& y, \
          Matrix<Real>& z, \
    const PDCOCtrl<Real>& ctrl );

#define EL_NO_INT_PROTO
#define EL_NO_COMPLEX_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include "El/macros/Instantiate.h"

} // namespace pdco
} // namespace El
