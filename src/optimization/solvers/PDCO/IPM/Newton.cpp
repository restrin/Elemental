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

    Output("========== Beginning Newton's Method ==========");

    // ========= Declarations of oft re-used variables ==============
    Int m = A.Height();
    Int n = A.Width();

    // Index sets to represent IR(ALL) and IR(0)
    vector<Int> ALL_m = IndexRange(m);
    vector<Int> ALL_n = IndexRange(n);
    vector<Int> ZERO (1,0);

    vector<Int> ixSetLow;  // Index set for lower bounded variables
    vector<Int> ixSetUpp;  // Index set for upper bounded variables
    vector<Int> ixSetFix;  // Index set for fixed bounded variables

    // For objective function
    Matrix<Real> grad;     // For gradient of phi
    Matrix<Real> Hess;     // For hessian of phi

    // Matrices used by KKT system
    Matrix<Real> bCopy;    // Modified b due to fixed variables
    Matrix<Real> D2sq;     // D2sq = D2^2
    Matrix<Real> At;       // A^T (after fixed variables removed)
    Matrix<Real> AtCopy;   // Copy of A^T
    Matrix<Real> ACopy;    // Used for KKT system
    Matrix<Real> H;        // Used for KKT system
    Matrix<Real> S;        // Schur complement

    // Various residuals and convergence measures
    Matrix<Real> w;        // Residual for KKT system
    Matrix<Real> r1;       // Primal residual
    Matrix<Real> r2;       // Dual residual
    Matrix<Real> cL;       // Lower bound complementarity residual
    Matrix<Real> cU;       // Upper bound complementarity residual
    Real center;           // Centering parameter
    Real Pfeas;            // Primal feasibility
    Real Dfeas;            // Dual feasibility
    Real Cfeas;            // Complementarity feasibility
    Real Cinf0;            // Complementarity convergence criteria

    // Artificial variables, step directions and step sizes
    Matrix<Real> xFix;
    Matrix<Real> z1;
    Matrix<Real> z2;
    Matrix<Real> dx;       // Primal step direction
    Matrix<Real> dy;       // Dual step direction
    Matrix<Real> dz1;      // Complementarity step direction
    Matrix<Real> dz2;      // Complementarity step direction
    Real stepx;            // Step size for x, y
    Real stepz;            // Step size for z1, z2
    Real stepmu;           // Step size for mu

    Real mulast = 0.1*ctrl.optTol; // Final value of mu
    Real mu = Max(ctrl.mu0,mulast);
    bool converged = false;

    // Miscellaneous
    Matrix<Real> zeros;    // Used to set submatrices to zero
    Matrix<Real> ones;     // Used to set diagonals to one
    bool diagHess = false; // Is the Hessian diagonal?

    // Initialize some useful variables
    Copy(D2, D2sq);
    DiagonalScale(LEFT, NORMAL, D2, D2sq);
    if( ctrl.method == Method::LDLy )
        Zeros(S, m, m);
    else if( ctrl.method == Method::LDLy )
        Zeros(S, n, n);
    // ==============================================================

    // Determine index sets for lower bounded variables,
    // upper bounded variables, and fixed variables
    pdco::ClassifyBounds(bl, bu, ixSetLow, ixSetUpp, ixSetFix, ctrl.print);

    Copy(b, bCopy);
    if( ixSetFix.size() > 0 )
    {
        // Fix b to allow for fixed variables
        Matrix<Real> Asub;
        GetSubmatrix(bl, ixSetFix, ZERO, xFix); // xFix = bl(ixSetFix)
        GetSubmatrix(A, ALL_m, ixSetFix, Asub); // Asub = A(:,ixSetFix)
        Gemv(NORMAL, Real(-1), Asub, xFix, Real(1), bCopy); // b = b - A*xFix
    }

    // Scale input data?

    // Initialize the data
    pdco::Initialize(x, y, z1, z2, bl, bu, 
      ixSetLow, ixSetUpp, ixSetFix, ctrl.x0min, ctrl.z0min, m, n, ctrl.print);

    // Compute residuals
    phi.grad(x, grad); // get gradient
    phi.hess(x, Hess); // get Hessian

    if( Hess.Width() == 1 ) // TODO: Better check?
      diagHess = true;

    ResidualPD(A, ixSetLow, ixSetUpp, ixSetFix,
      bCopy, D1, D2, grad, x, y, z1, z2, r1, r2);

    ResidualC(mu, ixSetLow, ixSetUpp, bl, bu, x, z1, z2, center, Cinf0, cL, cU);

    Pfeas = InfinityNorm(r1);
    Dfeas = InfinityNorm(r2);
    Cfeas = Max(InfinityNorm(cL), InfinityNorm(cU));

    if( ctrl.print )
    {
        Output("Initial feasibility: ");
        Output("  Pfeas  = ", Pfeas);
        Output("  Dfeas  = ", Dfeas);
        Output("  Cinf0  = ", Cinf0);
        Output("  ||cL||oo = ", InfinityNorm(cL));
        Output("  ||cU||oo = ", InfinityNorm(cU));
        Output("  center = ", center);
    }

    Copy(A, ACopy); // ACopy = A
    // Zero columns corresponding to fixed variables
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
        if( ctrl.print )
        {
            Output("======== Iteration: ", numIts, " ========");
            Output("  mu = ", mu);
        }
        switch( ctrl.method )
        {
            case Method::LDLy:
            {
                // We solve the system
                // [H   -A'] [dx] = [w ]
                // [A  D2^2] [dy]   [r2]
                // using the Schur complement to compute dy first

                FormHandW( Hess, D1, x, z1, z2, bl, bu,
                  ixSetLow, ixSetUpp, ixSetFix, r2, cL, cU, H, w, diagHess );

                if( ixSetFix.size() > 0 )
                {
                    // Set rows/cols corresponding to fixed variables to zero
                    Zeros(zeros, ixSetFix.size(), n);
                    Ones(ones, ixSetFix.size(), 1);
                    SetSubmatrix(H, ixSetFix, ALL_n, zeros);
                    Zeros(zeros, n, ixSetFix.size());
                    SetSubmatrix(H, ALL_n, ixSetFix, zeros);
                    // Fix diagonal to 1
                    UpdateSubdiagonal(H, ixSetFix, Real(1), ones);
                }

                // Make a copy of At
                Copy(At, AtCopy);

                // Start solving for dx, dy
                // TODO: Use pivoted?
                if( !diagHess )
                {
                  // Hessian is dense, need LDL and solve
                  LDL(H, false);
                  ldl::SolveAfter(H, AtCopy, false); // AtCopy = H\A'
                }
                else
                {
                  // Diagonal Hessian, use diagonal solve
                  DiagonalSolve(LEFT, NORMAL, H, AtCopy);
                }

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
            }
                break;
            case Method::LDLx:
                RuntimeError("LDLx not yet implemented.");
                break;
            case Method::LDL2:
                RuntimeError("LDL2 not yet implemented.");
                break;
            case Method::LDL25:
                RuntimeError("LDL25 not yet implemented.");
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
        converged = (Pfeas <= ctrl.feaTol)
          && (Dfeas <= ctrl.feaTol)
          && (Cinf0 <= ctrl.optTol);

        if( ctrl.print )
        {
            Output("  Pfeas    = ", Pfeas);
            Output("  Dfeas    = ", Dfeas);
            Output("  Cinf0    = ", Cinf0);
            Output("  ||cL||oo = ", InfinityNorm(cL));
            Output("  ||cU||oo = ", InfinityNorm(cU));
            Output("  center   = ", center);
        }

        if( ctrl.adaptiveMu )
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
    if( ixSetFix.size() > 0 )
    {
        GetSubmatrix(bl, ixSetFix, ZERO, xFix);
        SetSubmatrix(x, ixSetFix, ZERO, xFix);
    }

    Int lowerActive;
    Int upperActive;
    // Report active constraints
    GetActiveConstraints(x, bl, bu, ixSetLow, ixSetUpp,
      lowerActive, upperActive);

    // Reconstruct z from z1 and z2
    Matrix<Real> tmp;
    Zeros(z, n, 1);
    UpdateSubmatrix(z, ixSetLow, ZERO, Real(1), z1);
    UpdateSubmatrix(z, ixSetUpp, ZERO, Real(-1), z2);
    Matrix<Real> zFix;
    GetSubmatrix(grad, ixSetFix, ZERO, zFix);
    GetSubmatrix(r2, ixSetFix, ZERO, tmp);
    Axpy(Real(-1), tmp, zFix);
    UpdateSubmatrix(z, ixSetFix, ZERO, Real(1), zFix);

    Output("========== Completed Newton's Method ==========");

    if( converged )
      Output("Result: Converged!");
    else
      Output("Result: Failed!");

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

template<typename Real>
void Newton
( const PDCOObj<Real>& phi,
  const SparseMatrix<Real>& A,
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

    Output("========== Beginning Sparse Newton's Method ==========");

    // ========= Declarations of oft re-used variables ==============
    Int m = A.Height();
    Int n = A.Width();

    const Real eps = limits::Epsilon<Real>();
    const Real deltaTmp = Pow(eps, Real(0.25));

    // Index sets to represent IR(ALL) and IR(0)
    vector<Int> ALL_m = IndexRange(m);
    vector<Int> ALL_n = IndexRange(n);
    vector<Int> ZERO (1,0);

    function<Real(Real)> sqrtFunc
    ( []( Real alpha ) { return Sqrt(alpha); } ); // Elementwise sqrt

    vector<Int> ixSetLow;  // Index set for lower bounded variables
    vector<Int> ixSetUpp;  // Index set for upper bounded variables
    vector<Int> ixSetFix;  // Index set for fixed bounded variables

    // For objective function
    Matrix<Real> grad;     // For gradient of phi
    SparseMatrix<Real> Hess;     // For hessian of phi

    // Matrices used by KKT system
    Matrix<Real> bCopy;    // Modified b due to fixed variables
    Matrix<Real> D2sq;     // D2sq = D2^2
    Matrix<Real> D1sq;     // D1sqp = D1^2
    SparseMatrix<Real> ACopy;    // Used for KKT system
    SparseMatrix<Real> H;        // Used for KKT system

    // Various residuals and convergence measures
    Matrix<Real> w;        // Residual for KKT system
    Matrix<Real> r1;       // Primal residual
    Matrix<Real> r2;       // Dual residual
    Matrix<Real> cL;       // Lower bound complementarity residual
    Matrix<Real> cU;       // Upper bound complementarity residual
    Real center;           // Centering parameter
    Real Pfeas;            // Primal feasibility
    Real Dfeas;            // Dual feasibility
    Real Cfeas;            // Complementarity feasibility
    Real Cinf0;            // Complementarity convergence criteria

    // Artificial variables, step directions and step sizes
    Matrix<Real> xFix;
    Matrix<Real> z1;
    Matrix<Real> z2;
    Matrix<Real> dx;       // Primal step direction
    Matrix<Real> dy;       // Dual step direction
    Matrix<Real> dz1;      // Complementarity step direction
    Matrix<Real> dz2;      // Complementarity step direction
    Real stepx;            // Step size for x, y
    Real stepz;            // Step size for z1, z2
    Real stepmu;           // Step size for mu

    // Variables to avoid recomputation
    Matrix<Real> xmbl;     // x-bl
    Matrix<Real> bumx;     // bu-x

    Real mulast = 0.1*ctrl.optTol; // Final value of mu
    Real mu = Max(ctrl.mu0,mulast);
    bool converged = false;

    // Miscellaneous
    Matrix<Real> zeros;    // Used to set submatrices to zero
    Matrix<Real> ones;     // Used to set diagonals to one

    // Initialize some useful variables
    Copy(D2, D2sq);
    DiagonalScale(LEFT, NORMAL, D2, D2sq);
    Copy(D1, D1sq);
    DiagonalScale(LEFT, NORMAL, D1, D1sq);
    // ==============================================================

    // Determine index sets for lower bounded variables,
    // upper bounded variables, and fixed variables
    pdco::ClassifyBounds(bl, bu, ixSetLow, ixSetUpp, ixSetFix, ctrl.print);

    Copy(b, bCopy);
    if( ixSetFix.size() > 0 )
    {
        // Fix b to allow for fixed variables
        SparseMatrix<Real> Asub;
        GetSubmatrix(bl, ixSetFix, IR(0), xFix); // xFix = bl(ixSetFix)
        GetSubmatrix(A, ALL, ixSetFix, Asub); // Asub = A(:,ixSetFix)
        Multiply(NORMAL, Real(-1), Asub, xFix, Real(1), bCopy); // b = b - A*xFix
    }

    // Scale input data?

    // Initialize the data
    pdco::Initialize(x, y, z1, z2, bl, bu, 
      ixSetLow, ixSetUpp, ixSetFix, ctrl.x0min, ctrl.z0min, m, n, ctrl.print);

    // Compute residuals
    phi.grad(x, grad); // get gradient
    phi.sparseHess(x, Hess); // get Hessian

    ResidualPD(A, ixSetLow, ixSetUpp, ixSetFix,
      bCopy, D1, D2, grad, x, y, z1, z2, r1, r2);

    ResidualC(mu, ixSetLow, ixSetUpp, bl, bu, x, z1, z2, center, Cinf0, cL, cU);

    Pfeas = InfinityNorm(r1);
    Dfeas = InfinityNorm(r2);
    Cfeas = Max(InfinityNorm(cL), InfinityNorm(cU));

    if( ctrl.print )
    {
        Output("Initial feasibility: ");
        Output("  Pfeas  = ", Pfeas);
        Output("  Dfeas  = ", Dfeas);
        Output("  Cinf0  = ", Cinf0);
        Output("  ||cL||oo = ", InfinityNorm(cL));
        Output("  ||cU||oo = ", InfinityNorm(cU));
        Output("  center = ", center);
    }

    ACopy = SparseMatrix<Real>(A); // ACopy = A
    // Zero columns corresponding to fixed variables
    if( ixSetFix.size() > 0 )
    {
        Zeros(zeros, ixSetFix.size(), n);
        auto Asub = ACopy(ALL, ixSetFix);
        Zeros(Asub, m, ixSetFix.size());
    }

    // Initialize static ortion of the KKT system
    vector<Int> map;
    vector<Int> invMap;
    ldl::Separator rootSep;
    ldl::NodeInfo info;
    SparseMatrix<Real> K;
    ldl::Front<Real> KFront;

    // Temporary regularization
    Matrix<Real> regTmp;
    Zeros(regTmp, n+m, 1);
//    const Real twoNormEstA = TwoNormEstimate( A, 6 );

    // Main loop
    for( Int numIts=0; numIts<=ctrl.maxIts; ++numIts )
    {
        if( ctrl.print )
        {
            Output("======== Iteration: ", numIts, " ========");
            Output("  mu = ", mu);
        }
        switch( ctrl.method )
        {
            case Method::LDLy:
                RuntimeError("LDLy not yet implemented.");
                break;
            case Method::LDLx:
                RuntimeError("LDLx not yet implemented.");
                break;
            case Method::LDL2:
            {
                // We solve the system
                // [H   A' ] [ dx] = [w ] = w // excuse the abuse of notation
                // [A -D2^2] [-dy]   [r2]
                // By solving the KKT system

                // Form the KKT system
                FormKKT( Hess, ACopy, D1sq, D2sq, x, z1, z2, bl, bu, 
                    ixSetLow, ixSetUpp, ixSetFix, K );
                // Form the right-hand side
                FormKKTRHS( x, r1, r2, cL, cU, bl, bu, ixSetLow, ixSetUpp, w );
                SparseMatrix<Real> KOrig(K);

                // Add temporary regularization for sparse LDL
//               Real MaxNormH = MaxNorm(Hess);
//               Ones(regTmp, n+m, 1);
//               auto regTmp1 = regTmp(IR(0,n), IR(0));
//               regTmp1 *= (twoNormEstA + MaxNormH + 1)*deltaTmp*deltaTmp;
//               auto regTmp2 = regTmp(IR(n,n+m), IR(0));
//               regTmp2 *= -(twoNormEstA + MaxNormH + 1)*deltaTmp*deltaTmp;

                UpdateDiagonal(K, Real(1), regTmp);

                if( numIts == 0 )
                {
                    // Get static nested dissection data
                    NestedDissection( K.LockedGraph(), map, rootSep, info );
                    InvertMap( map, invMap );
                }

                KFront.Pull( K, map, info );
                LDL( info, KFront, LDL_2D );

//                ldl::SolveAfter( invMap, info, KFront, w );
                reg_ldl::SolveAfter( KOrig, regTmp, invMap, info, KFront, w, ctrl.solveCtrl );

                dx = w( IR(0,n), IR(0) );
                dy = w( IR(n,END), IR(0) );
                dy *= -1;

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
            }
                break;
            case Method::LDL25:
            {
                // We solve the system
                // [H   A' ] [ dx] = [w ] = w // excuse the abuse of notation
                // [A -D2^2] [-dy]   [r2]
                // By symmetrically 'preconditioning' it with
                // [(x-bl)(bu-x) 0]^(1/2)
                // [ 0           I]

                Copy(x, xmbl);
                xmbl -= bl;
                Copy(bu, bumx);
                bumx -= x;

                // Form the KKT system
                FormKKT25( Hess, ACopy, D1sq, D2sq, x, z1, z2, bl, bu, 
                    xmbl, bumx, ixSetLow, ixSetUpp, ixSetFix, K );

                // NOTE: FormKKT25 takes square root of xmbl, bumx
                // They are now to be treated as the square roots

                // Only need square root
                // TODO: Deal with garbage entries?
//                EntrywiseMap(xmbl, sqrtFunc);
//                EntrywiseMap(bumx, sqrtFunc);

                FormKKTRHS25( x, r1, r2, cL, cU, bl, bu, 
                    xmbl, bumx, ixSetLow, ixSetUpp, w );
                SparseMatrix<Real> KOrig(K);

                UpdateDiagonal(K, Real(1), regTmp);

                if( numIts == 0 )
                {
                    // Get static nested dissection data
                    NestedDissection( K.LockedGraph(), map, rootSep, info );
                    InvertMap( map, invMap );
                }

                KFront.Pull( K, map, info );
                LDL( info, KFront, LDL_2D );

//                ldl::SolveAfter( invMap, info, KFront, w );
                reg_ldl::SolveAfter( KOrig, regTmp, invMap, info, KFront, w, ctrl.solveCtrl );

                dy = w( IR(n,END), IR(0) );
                dy *= -1;
                dx = w( IR(0,n), IR(0) );

                Matrix<Real> dxSub;
                Matrix<Real> xbSub;

                // Undo scaling on dx
                GetSubmatrix(xmbl, ixSetLow, ZERO, xbSub);
                GetSubmatrix(dx, ixSetLow, ZERO, dxSub);
                DiagonalScale(LEFT, NORMAL, xbSub, dxSub);
                SetSubmatrix(dx, ixSetLow, ZERO, dxSub);

                GetSubmatrix(bumx, ixSetUpp, ZERO, xbSub);
                GetSubmatrix(dx, ixSetUpp, ZERO, dxSub);
                DiagonalScale(LEFT, NORMAL, xbSub, dxSub);
                SetSubmatrix(dx, ixSetUpp, ZERO, dxSub);

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
            }
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
        converged = (Pfeas <= ctrl.feaTol)
          && (Dfeas <= ctrl.feaTol)
          && (Cinf0 <= ctrl.optTol);

        if( ctrl.print )
        {
            Output("  Pfeas    = ", Pfeas);
            Output("  Dfeas    = ", Dfeas);
            Output("  Cinf0    = ", Cinf0);
            Output("  ||cL||oo = ", InfinityNorm(cL));
            Output("  ||cU||oo = ", InfinityNorm(cU));
            Output("  center   = ", center);
        }

        if( ctrl.adaptiveMu )
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
        phi.sparseHess(x, Hess);

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
    if( ixSetFix.size() > 0 )
    {
        GetSubmatrix(bl, ixSetFix, ZERO, xFix);
        SetSubmatrix(x, ixSetFix, ZERO, xFix);
    }

    Int lowerActive;
    Int upperActive;
    // Report active constraints
    GetActiveConstraints(x, bl, bu, ixSetLow, ixSetUpp,
      lowerActive, upperActive);

    // Reconstruct z from z1 and z2
    Matrix<Real> tmp;
    Zeros(z, n, 1);
    UpdateSubmatrix(z, ixSetLow, ZERO, Real(1), z1);
    UpdateSubmatrix(z, ixSetUpp, ZERO, Real(-1), z2);
    Matrix<Real> zFix;
    GetSubmatrix(grad, ixSetFix, ZERO, zFix);
    GetSubmatrix(r2, ixSetFix, ZERO, tmp);
    Axpy(Real(-1), tmp, zFix);
    UpdateSubmatrix(z, ixSetFix, ZERO, Real(1), zFix);

    Output("========== Completed Newton's Method ==========");

    if( converged )
      Output("Result: Converged!");
    else
      Output("Result: Failed!");

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
    const PDCOCtrl<Real>& ctrl ); \
  template void Newton \
  ( const pdco::PDCOObj<Real>& phi, \
    const SparseMatrix<Real>& A, \
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
