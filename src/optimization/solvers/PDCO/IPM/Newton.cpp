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

static Timer totalTimer, ldlTimer,
             kktTimer, kktrhsTimer,
             hessTimer, gradTimer,
             solveAfterTimer;

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
        Matrix<Real>  bl,
        Matrix<Real>  bu,
  const Matrix<Real>& D1,
  const Matrix<Real>& D2,
        Matrix<Real>& x,
        Matrix<Real>& r,
        Matrix<Real>& y,
        Matrix<Real>& z,
  const PDCOCtrl<Real>& ctrl )
{
    EL_DEBUG_CSE

    Output("========== Beginning Newton's Method ==========");

    // ========= Declarations of oft re-used variables ==============
    Int m = A.Height();
    Int n = A.Width();

    const double log10 = Log(double(10));

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
    Matrix<Real> D1sq;     // D1sq = D1^2
    Matrix<Real> D2sq;     // D2sq = D2^2
    Matrix<Real> At;       // A^T (after fixed variables removed)
    Matrix<Real> AtCopy;   // Copy of A^T
    Matrix<Real> ACopy;    // Used for KKT system
    Matrix<Real> H;        // Used for KKT system
    Matrix<Real> S;        // Schur complement
    Matrix<Real> dRow;     // Row scaling when equilibrating
    Matrix<Real> dCol;     // Column scaling when equilibrating

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
    Matrix<Real> xin;      // Scaled x variable for input to phi

    // For scaling purposes
    Real beta = 1;
    Real zeta = 1;
    Real theta = 1;

    Real mulast = 0.1*ctrl.optTol; // Final value of mu
    Real mu = Max(ctrl.mu0,mulast);
    bool converged = false;

    // Miscellaneous
    Matrix<Real> zeros;    // Used to set submatrices to zero
    Matrix<Real> ones;     // Used to set diagonals to one
    bool diagHess = false; // Is the Hessian diagonal?

    // Boolean flag to check if variables already initialized
    bool initialized = false;

    // Initialize some useful variables
    Copy(D2, D2sq);
    DiagonalScale(LEFT, NORMAL, D2, D2sq);
    Copy(D1, D1sq);
    DiagonalScale(LEFT, NORMAL, D1, D1sq);

    if( ctrl.method == Method::LDLy )
        Zeros(S, m, m);
    else if( ctrl.method == Method::LDLy )
        Zeros(S, n, n);
    // ==============================================================

    // ======= Begin initialization stuff =======
    // Determine index sets for lower bounded variables,
    // upper bounded variables, and fixed variables
    pdco::ClassifyBounds(bl, bu, ixSetLow, ixSetUpp, ixSetFix, ctrl.print);

    // Ensure that all variables are initialized or none of them are
    if( x.Height() > 0 || y.Height() > 0 || z.Height() > 0 )
    {
        pdco::CheckVariableInit(x, y, z, bl, bu, ixSetLow, ixSetUpp, ixSetFix);
        initialized = true;
    }

    Copy(b, bCopy);
    Copy(A, ACopy); // ACopy = A
    // Zero columns corresponding to fixed variables
    if( ixSetFix.size() > 0 )
    {
        // Fix b to allow for fixed variables
        Matrix<Real> Asub;
        GetSubmatrix(bl, ixSetFix, ZERO, xFix); // xFix = bl(ixSetFix)
        GetSubmatrix(A, ALL_m, ixSetFix, Asub); // Asub = A(:,ixSetFix)
        Gemv(NORMAL, Real(-1), Asub, xFix, Real(1), bCopy); // b = b - A*xFix

        Zeros(zeros, ixSetFix.size(), n);
        SetSubmatrix(ACopy, ALL_m, ixSetFix, zeros);
    }

    // Equilibrate the A matrix
    // TODO: Adjust feas- and opt-tol?
    if( ctrl.outerEquil )
    {
        GeomEquil( ACopy, dRow, dCol, ctrl.print );
        DiagonalSolve( LEFT, NORMAL, dRow, bCopy );

        // Fix the bounds
        DiagonalScale( LEFT, NORMAL, dCol, bl );
        DiagonalScale( LEFT, NORMAL, dCol, bu );
    }

    // Scale input data
    if( ctrl.scale )
    {
        beta = Max(InfinityNorm(b), Real(1));

        if( !initialized )
        {
            // Initialize to get feasible point for gradient estimate
            pdco::Initialize(x, y, z1, z2, bl, bu, 
              ixSetLow, ixSetUpp, ixSetFix, ctrl.x0min, ctrl.z0min, m, n, ctrl.print);
        }
        if( ctrl.outerEquil )
        {
            DiagonalSolve( LEFT, NORMAL, dCol, xin );
            phi.grad( x, grad ); // get gradient
            DiagonalSolve( LEFT, NORMAL, dCol, grad );            
        }
        else
            phi.grad( x, grad ); // get gradient

        zeta = Max(InfinityNorm(grad),Real(1));

        theta = beta*zeta;

        bl *= Real(1)/beta;
        bu *= Real(1)/beta;
        bCopy *= Real(1)/beta;

        D1sq *= beta*beta/(theta);
        D2sq *= theta/(beta*beta);
    }

    if( ctrl.print )
    {
        Output("  beta = ", beta);
        Output("  zeta = ", zeta);
    }

    if( !initialized )
    {
        // Initialize the data if required
        pdco::Initialize(x, y, z1, z2, bl, bu, 
          ixSetLow, ixSetUpp, ixSetFix, ctrl.x0min, ctrl.z0min, m, n, ctrl.print);
    }
    else
    {
        pdco::Getz1z2(z, ixSetLow, ixSetUpp, z1, z2);
        // Scale the data
        x *= Real(1)/beta;
        y *= Real(1)/zeta;
        z1 *= Real(1)/zeta;
        z2 *= Real(1)/zeta;

        if( ctrl.outerEquil )
        {
            DiagonalScale( LEFT, NORMAL, dCol, x );
            DiagonalScale( LEFT, NORMAL, dRow, y );
            DiagonalSolve( LEFT, NORMAL, dCol, z1 );
            DiagonalSolve( LEFT, NORMAL, dCol, z2 );            
        }
    }
    //==== End of Initialization stuff =====

    // Compute residuals
    Copy(x, xin);
    xin *= beta;
    if( ctrl.outerEquil )
    {
        DiagonalSolve( LEFT, NORMAL, dCol, xin );
        phi.grad( xin, grad ); // get gradient
        phi.hess( xin, Hess ); // get Hessian
        DiagonalSolve( LEFT, NORMAL, dCol, grad );
        SymmetricDiagonalSolve( dCol, Hess );
    }
    else
    {
        phi.grad(xin, grad); // get gradient
        phi.hess(xin, Hess); // get Hessian
    }
    grad *= beta/theta;
    Hess *= beta*beta/theta;

    if( Hess.Width() == 1 ) // TODO: Better check?
      diagHess = true;

    ResidualPD(A, ixSetLow, ixSetUpp, ixSetFix,
      bCopy, D1sq, D2sq, grad, x, y, z1, z2, r1, r2);

    ResidualC(mu, ixSetLow, ixSetUpp, bl, bu, x, z1, z2, center, Cinf0, cL, cU);

    Pfeas = InfinityNorm(r1);
    Dfeas = InfinityNorm(r2);
    Cfeas = Max(InfinityNorm(cL), InfinityNorm(cU));

    if( ctrl.print )
    {
        Output("Iter\tmu\tPfeas\tDfeas\tCinf0\t|cL|oo\t|cU|oo\tcenter\tstepx\tstepz");
        printf("Init :\t%1.1f\t%1.1f\t%1.1f\t%1.1f\t%1.1f\t%1.1f\t%1.1f\n", 
            Log(double(mu))/log10, Log(double(Pfeas))/log10,
            Log(double(Dfeas))/log10, Log(double(Cinf0))/log10, 
            Log(double(InfinityNorm(cL)))/log10, Log(double(InfinityNorm(cU)))/log10,
            Log(double(center))/log10);
    }

    // Get transpose
    Transpose(ACopy, At);

    // Main loop
    for( Int numIts=0; numIts<=ctrl.maxIts; ++numIts )
    {
        switch( ctrl.method )
        {
            case Method::LDLy:
            {
                // We solve the system
                // [H   -A'] [dx] = [w ]
                // [A  D2^2] [dy]   [r2]
                // using the Schur complement to compute dy first

                FormHandW( Hess, D1sq, x, z1, z2, bl, bu,
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
        bool success = Linesearch(phi, mu, ACopy, bCopy, bl, bu, D1sq, D2sq, 
          x, y, z1, z2, r1, r2, center, Cinf0, cL, cU, stepx, stepz,
          dx, dy, dz1, dz2, ixSetLow, ixSetUpp, ixSetFix, dCol, beta, theta, ctrl);

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
            printf("%d :\t%1.1f\t%1.1f\t%1.1f\t%1.1f\t%1.1f\t%1.1f\t%1.1f\t%1.3f\t%1.3f\n", 
                numIts, Log(double(mu))/log10, Log(double(Pfeas))/log10,
                Log(double(Dfeas))/log10, Log(double(Cinf0))/log10, 
                Log(double(InfinityNorm(cL)))/log10, Log(double(InfinityNorm(cU)))/log10,
                Log(double(center))/log10, stepx, stepz);
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
        Copy(x, xin);
        xin *= beta;
        if( ctrl.outerEquil )
        {
            DiagonalSolve( LEFT, NORMAL, dCol, xin );
            phi.grad( xin, grad ); // get gradient
            phi.hess( xin, Hess ); // get Hessian
            DiagonalSolve( LEFT, NORMAL, dCol, grad );
            SymmetricDiagonalSolve( dCol, Hess );
        }
        else
        {
            phi.grad(xin, grad); // get gradient
            phi.hess(xin, Hess); // get Hessian
        }
        grad *= beta/theta;
        Hess *= beta*beta/theta;

        // Recompute residuals
        ResidualPD(ACopy, ixSetLow, ixSetUpp, ixSetFix,
          bCopy, D1sq, D2sq, grad, x, y, z1, z2, r1, r2);

        ResidualC(mu, ixSetLow, ixSetUpp, bl, bu, x, z1, z2, center, Cinf0, cL, cU);

        if (converged)
            break;
    } // end main for loop

    // Undo scaling by beta and zeta
    if( ctrl.scale )
    {
        x *= beta;
        y *= zeta;
        z1 *= zeta;
        z2 *= zeta;

        bl *= beta;
        bu *= beta;
    }

    // Undo scaling due to equilibration
    if( ctrl.outerEquil )
    {
        DiagonalSolve( LEFT, NORMAL, dCol, x );
        DiagonalSolve( LEFT, NORMAL, dRow, y );

        Matrix<Real> tmp;
        GetSubmatrix(dCol, ixSetLow, ZERO, tmp);
        DiagonalScale( LEFT, NORMAL, tmp, z1 );
        GetSubmatrix(dCol, ixSetUpp, ZERO, tmp);
        DiagonalScale( LEFT, NORMAL, tmp, z2 );

        DiagonalSolve( LEFT, NORMAL, dCol, bl );
        DiagonalSolve( LEFT, NORMAL, dCol, bu );
    }

    Int lowerActive;
    Int upperActive;
    // Report active constraints
    GetActiveConstraints(x, bl, bu, ixSetLow, ixSetUpp,
      lowerActive, upperActive);

    Output("========== Completed Newton's Method ==========");

    if( converged )
      Output("Result: Converged!");
    else
      Output("Result: Failed!");

    Copy(D2, D2sq);
    Copy(D1, D1sq);
    if( ctrl.outerEquil )
    {
        DiagonalScale(LEFT, NORMAL, dRow, D2sq);
        DiagonalScale(LEFT, NORMAL, dCol, D1sq);
    }
    DiagonalScale(LEFT, NORMAL, D2sq, D2sq);
    DiagonalScale(LEFT, NORMAL, D1sq, D1sq);

    // Reconstruct z from z1 and z2
    phi.grad( x, grad ); // get gradient
    ResidualPD(A, ixSetLow, ixSetUpp, IndexRange(0),
        b, D1sq, D2sq, grad, x, y, z1, z2, r1, r2);
    Matrix<Real> tmp;
    Zeros(z, n, 1);
    UpdateSubmatrix(z, ixSetLow, ZERO, Real(1), z1);
    UpdateSubmatrix(z, ixSetUpp, ZERO, Real(-1), z2);
    Matrix<Real> zFix;
    GetSubmatrix(r2, ixSetFix, ZERO, zFix);
    SetSubmatrix(z, ixSetFix, ZERO, zFix);

    // Compute unscaled residuals
    // TODO: Avoid recomputing some of this stuff
    ResidualPD(A, b, D1sq, D2sq, grad, x, y, z, r1, r2);
    ResidualC(mu, ixSetLow, ixSetUpp, bl, bu, x, z1, z2, center, Cinf0, cL, cU);

    Pfeas = InfinityNorm(r1);
    Dfeas = InfinityNorm(r2);
    Real cLInf = InfinityNorm(cL);
    Real cUInf = InfinityNorm(cU);
    Cfeas = Max(cLInf, cUInf);

    Real xInf = InfinityNorm(x);
    Real zInf = InfinityNorm(z);
    Output("  Pfeas         = ", Pfeas);
    Output("  Dfeas         = ", Dfeas);
    Output("  Cinf0         = ", Cinf0);
    Output("  |cL|oo        = ", cLInf);
    Output("  |cU|oo        = ", cUInf);
    Output("  Cinf0/|x||z|  = ", Cinf0/(xInf * zInf));
    Output("  |cL|oo/|x||z| = ", cLInf/(xInf * zInf));
    Output("  |cU|oo/|x||z| = ", cUInf/(xInf * zInf));
    Output("  center        = ", center);

    Output();
    Output("  Scaled:   max |x| = ", Max(x)/beta, "\tmax |y| = ", Max(y)/zeta, "\tmax |z| = ", Max(z)/zeta);
    Output("  Unscaled: max |x| = ", Max(x), "\tmax |y| = ", Max(y), "\tmax |z| = ", Max(z));
    Output();

    Output("  Number active constraints on");
    Output("    Lower bound: ", lowerActive);
    Output("    Upper bound: ", upperActive);
}

template<typename Real>
void Newton
( const PDCOObj<Real>& phi,
  const SparseMatrix<Real>& A,
  const Matrix<Real>& b, 
        Matrix<Real>  bl,
        Matrix<Real>  bu,
  const Matrix<Real>& D1,
  const Matrix<Real>& D2,
        Matrix<Real>& x,
        Matrix<Real>& r,
        Matrix<Real>& y,
        Matrix<Real>& z,
  const PDCOCtrl<Real>& ctrl )
{
    EL_DEBUG_CSE

    Output("========== Beginning Sparse Newton's Method ==========");

    if( ctrl.time )
    {
        totalTimer.Reset();
        ldlTimer.Reset();
        kktTimer.Reset();
        kktrhsTimer.Reset();
        hessTimer.Reset();
        gradTimer.Reset();
        solveAfterTimer.Reset();

        totalTimer.Start();
    }

    // ========= Declarations of oft re-used variables ==============
    Int m = A.Height();
    Int n = A.Width();

    const Real eps = limits::Epsilon<Real>();
    const Real deltaTmp = ctrl.deltaTmp;
    const double log10 = Log(double(10));

    // Index sets to represent IR(ALL) and IR(0)
    vector<Int> ALL_m = IndexRange(m);
    vector<Int> ALL_n = IndexRange(n);
    vector<Int> ZERO (1,0);

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
    Matrix<Real> dRow;     // Row scaling when equilibrating
    Matrix<Real> dCol;     // Column scaling when equilibrating

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
    Matrix<Real> xin;      // Scaled x variable for input to phi

    // For scaling purposes
    Real beta = Real(1);
    Real zeta = Real(1);
    Real theta = Real(1);

    // Variables to avoid recomputation
    Matrix<Real> xmbl;     // x-bl
    Matrix<Real> bumx;     // bu-x

    Real mulast = 0.1*ctrl.optTol; // Final value of mu
    Real mu = Max(ctrl.mu0,mulast);
    bool converged = false;

    // Miscellaneous
    Matrix<Real> zeros;    // Used to set submatrices to zero
    Matrix<Real> ones;     // Used to set diagonals to one

    // Boolean flag to check if variables already initialized
    bool initialized = false;

    // Initialize some useful variables
    Copy(D2, D2sq);
    DiagonalScale(LEFT, NORMAL, D2, D2sq);
    Copy(D1, D1sq);
    DiagonalScale(LEFT, NORMAL, D1, D1sq);

    Ones( xmbl, n, 1 );
    Ones( bumx, n, 1 );

    Zeros( xin, n, 1 );
    // ==============================================================

    // ======= Begin initialization stuff =======

    // Determine index sets for lower bounded variables,
    // upper bounded variables, and fixed variables
    pdco::ClassifyBounds(bl, bu, ixSetLow, ixSetUpp, ixSetFix, ctrl.print);

    // Ensure that all variables are initialized or none of them are
    if( x.Height() > 0 || y.Height() > 0 || z.Height() > 0 )
    {
        pdco::CheckVariableInit(x, y, z, bl, bu, ixSetLow, ixSetUpp, ixSetFix);
        initialized = true;
    }

    Copy(b, bCopy);
    ACopy = SparseMatrix<Real>(A); // ACopy = A
    if( ixSetFix.size() > 0 )
    {
        Matrix<Real> xFix;
        Zeros(xFix, n, 1);
        GetSubmatrix(bl, ixSetFix, ZERO, xFix);
        Multiply(NORMAL, Real(-1), ACopy, xFix, Real(1), bCopy);

        ZeroSubmatrix(ACopy, ALL_m, ixSetFix);
    }

    // Equilibrate the A matrix
    // TODO: Adjust feas- and opt-tol?
    if( ctrl.outerEquil )
    {
        GeomEquil( ACopy, dRow, dCol, ctrl.print );
        // Ones(dCol, n, 1);
        // dCol *= Real(100);
        // Ones(dRow, m, 1);
        // // dRow *= Real(1)/Real(100);
        // ACopy *= Real(1)/Real(100);

        DiagonalSolve( LEFT, NORMAL, dRow, bCopy );

        // Fix the bounds
        DiagonalScale( LEFT, NORMAL, dCol, bl );
        DiagonalScale( LEFT, NORMAL, dCol, bu );
    }

    // Scale input data
    if( ctrl.scale )
    {
        beta = Max(InfinityNorm(bCopy), Real(1));

        if( !initialized )
        {
            // Initialize to get feasible point for gradient estimate
            pdco::Initialize(x, y, z1, z2, bl, bu, 
              ixSetLow, ixSetUpp, ixSetFix, ctrl.x0min, ctrl.z0min, m, n, ctrl.print);
        }

        if( ctrl.outerEquil )
        {
            DiagonalSolve( LEFT, NORMAL, dCol, xin );
            phi.grad( x, grad ); // get gradient
            DiagonalSolve( LEFT, NORMAL, dCol, grad );            
        }
        else
            phi.grad( x, grad ); // get gradient

        zeta = Max(InfinityNorm(grad),Real(1));

        theta = beta*zeta;

        bl *= Real(1)/beta;
        bu *= Real(1)/beta;
        bCopy *= Real(1)/beta;

        D1sq *= beta*beta/(theta);
        D2sq *= theta/(beta*beta);
    }

    if( ctrl.print )
    {
        Output("  beta = ", beta);
        Output("  zeta = ", zeta);
    }

    if( !initialized )
    {
        // Initialize the data if required
        pdco::Initialize(x, y, z1, z2, bl, bu, 
          ixSetLow, ixSetUpp, ixSetFix, ctrl.x0min, ctrl.z0min, m, n, ctrl.print);
    }
    else
    {
        pdco::Getz1z2(z, ixSetLow, ixSetUpp, z1, z2);
        // Scale the data
        x *= Real(1)/beta;
        y *= Real(1)/zeta;
        z1 *= Real(1)/zeta;
        z2 *= Real(1)/zeta;

        if( ctrl.outerEquil )
        {
            DiagonalScale( LEFT, NORMAL, dCol, x );
            DiagonalScale( LEFT, NORMAL, dRow, y );
            DiagonalSolve( LEFT, NORMAL, dCol, z1 );
            DiagonalSolve( LEFT, NORMAL, dCol, z2 );            
        }
    }
    //==== End of Initialization stuff =====

    Copy(x, xin);
    xin *= beta;
    if( ctrl.outerEquil )
    {
        DiagonalSolve( LEFT, NORMAL, dCol, xin );
        if( ctrl.time )
            gradTimer.Start();
        phi.grad( xin, grad ); // get gradient
        if( ctrl.time )
        {
            gradTimer.Stop();
            hessTimer.Start();
        }
        phi.sparseHess( xin, Hess ); // get Hessian
        if( ctrl.time )
            hessTimer.Stop();
        DiagonalSolve( LEFT, NORMAL, dCol, grad );
        SymmetricDiagonalSolve( dCol, Hess );
    }
    else
    {
        if( ctrl.time )
            gradTimer.Start();
        phi.grad( xin, grad ); // get gradient
        if( ctrl.time )
        {
            gradTimer.Stop();
            hessTimer.Start();
        }
        phi.sparseHess( xin, Hess ); // get Hessian
        if( ctrl.time )
            hessTimer.Stop();
    }
    grad *= beta/theta;
    Hess *= beta*beta/theta;

    // Compute residuals
    ResidualPD(ACopy, ixSetLow, ixSetUpp, ixSetFix,
      bCopy, D1sq, D2sq, grad, x, y, z1, z2, r1, r2);

    ResidualC(mu, ixSetLow, ixSetUpp, bl, bu, x, z1, z2, center, Cinf0, cL, cU);

    Pfeas = InfinityNorm(r1);
    Dfeas = InfinityNorm(r2);
    Cfeas = Max(InfinityNorm(cL), InfinityNorm(cU));

    if( ctrl.print )
    {
        Output("Iter\tmu\tPfeas\tDfeas\tCinf0\t|cL|oo\t|cU|oo\tcenter\tstepx\tstepz");
        printf("Init :\t%1.1f\t%1.1f\t%1.1f\t%1.1f\t%1.1f\t%1.1f\t%1.1f\n", 
            Log(double(mu))/log10, Log(double(Pfeas))/log10,
            Log(double(Dfeas))/log10, Log(double(Cinf0))/log10, 
            Log(double(InfinityNorm(cL)))/log10, Log(double(InfinityNorm(cU)))/log10,
            Log(double(center))/log10);
    }

    // Initialize static portion of the KKT system
    vector<Int> map;
    vector<Int> invMap;
    ldl::Separator rootSep;
    ldl::NodeInfo info;
    SparseMatrix<Real> K, KOrig;
    SparseLDLFactorization<Real> sparseLDLFact;
    BisectCtrl bisectCtrl = BisectCtrl();
    bisectCtrl.cutoff = 128; // TODO: Make tunable?

    // Temporary regularization
    Matrix<Real> regTmp;
    Zeros(regTmp, n+m, 1);
    const Real twoNormEstA = MaxNorm(ACopy);
    //const Real twoNormEstA = TwoNormEstimate( ACopy, 6 );

    // Main loop
    for( Int numIts=0; numIts<=ctrl.maxIts; ++numIts )
    {
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
                // [A -D2^2] [-dy]   [r1]
                // By solving the KKT system

                // Add temporary regularization for sparse LDL
                Real MaxNormH = MaxNorm(Hess);
                Ones(regTmp, n+m, 1);
                auto regTmp1 = regTmp(IR(0,n), IR(0));
                regTmp1 *= (twoNormEstA + MaxNormH + 1)*deltaTmp*deltaTmp;
                auto regTmp2 = regTmp(IR(n,n+m), IR(0));
                regTmp2 *= -(twoNormEstA + MaxNormH + 1)*deltaTmp*deltaTmp;

                if( ctrl.time )
                    kktTimer.Start();
                // Form the KKT system
                FormKKT( Hess, ACopy, D1sq, D2sq, x, z1, z2, bl, bu, 
                    ixSetLow, ixSetUpp, ixSetFix, K );
                if( ctrl.time )
                    kktTimer.Stop();

                if( ctrl.time )
                    kktrhsTimer.Start();
                // Form the right-hand side
                FormKKTRHS( x, r1, r2, cL, cU, bl, bu, ixSetLow, ixSetUpp, w );
                if( ctrl.time )
                    kktrhsTimer.Stop();
                KOrig = K;

                UpdateDiagonal(K, Real(1), regTmp);

                if( numIts == 0 )
                {
                    // Get static nested dissection data
                    const bool hermitian = true;
                    sparseLDLFact.Initialize( K, hermitian, bisectCtrl );

                }
                else
                    sparseLDLFact.ChangeNonzeroValues( K );
 
                if( ctrl.time )
                    ldlTimer.Start();
                sparseLDLFact.Factor( LDL_2D );
                if( ctrl.time )
                    ldlTimer.Stop();

                if( ctrl.time )
                    solveAfterTimer.Start();
                reg_ldl::SolveAfter( KOrig, regTmp, sparseLDLFact, w, ctrl.solveCtrl );
                if( ctrl.time )
                    solveAfterTimer.Stop();

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
                DiagonalSolve(LEFT, NORMAL, tmp2, dz2); // dz2 = (bu-x)^-1 * (cU + z2*dx)
            }
                break;
            case Method::LDL25:
            {
                // We solve the system
                // [H   A' ] [ dx] = [w ] = w // excuse the abuse of notation
                // [A -D2^2] [-dy]   [r1]
                // By symmetrically 'preconditioning' it with
                // [(x-bl)(bu-x) 0]^(1/2)
                // [ 0           I]

                // Form (x-bl) and (bu-x)
                Matrix<Real> xmbSub;
                Matrix<Real> tmp;
                Copy(x, xmbSub);
                xmbSub -= bl;
                GetSubmatrix( xmbSub, ixSetLow, ZERO, tmp );
                SetSubmatrix( xmbl, ixSetLow, ZERO, tmp );
                Copy(bu, xmbSub);
                xmbSub -= x;
                GetSubmatrix( xmbSub, ixSetUpp, ZERO, tmp );
                SetSubmatrix( bumx, ixSetUpp, ZERO, tmp );

                // Add temporary regularization
                Real MaxNormH = MaxNorm(Hess);
                Ones(regTmp, n+m, 1);
                auto regTmp1 = regTmp(IR(0,n), IR(0));
                regTmp1 *= (twoNormEstA + MaxNormH + 1)*deltaTmp*deltaTmp;
                auto regTmp2 = regTmp(IR(n,n+m), IR(0));
                regTmp2 *= -(twoNormEstA + MaxNormH + 1)*deltaTmp*deltaTmp;

                if( ctrl.time )
                    kktTimer.Start();
                // Form the KKT system
                FormKKT25( Hess, ACopy, D1sq, D2sq, x, z1, z2, bl, bu, 
                    xmbl, bumx, ixSetLow, ixSetUpp, ixSetFix, K );
                if( ctrl.time )
                    kktTimer.Stop();

                // NOTE: FormKKT25 takes square root of xmbl, bumx
                // They are now to be treated as the square roots
                // TODO: Deal with garbage entries due to infs?

                if( ctrl.time )
                    kktrhsTimer.Start();
                FormKKTRHS25( x, r1, r2, cL, cU, bl, bu, 
                    xmbl, bumx, ixSetLow, ixSetUpp, w );
                if( ctrl.time )
                    kktrhsTimer.Stop();
                KOrig = K;

                UpdateDiagonal(K, Real(1), regTmp);

                if( numIts == 0 )
                {
                    // Get static nested dissection data
                    const bool hermitian = true;
                    sparseLDLFact.Initialize( K, hermitian, bisectCtrl );
                }
                else
                    sparseLDLFact.ChangeNonzeroValues( K );

                if( ctrl.time )
                    ldlTimer.Start();
                sparseLDLFact.Factor( LDL_2D );
                if( ctrl.time )
                    ldlTimer.Stop();

                if( ctrl.time )
                    solveAfterTimer.Start();
                reg_ldl::SolveAfter( KOrig, regTmp, sparseLDLFact, w, ctrl.solveCtrl );
                if( ctrl.time )
                    solveAfterTimer.Stop();

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
                DiagonalSolve(LEFT, NORMAL, tmp2, dz1); // dz2 = (x-bl)^-1 * (cL - z1*dx)

                // Compute dz2
                Copy(bu, tmp1);
                tmp1 -= x;
                GetSubmatrix(tmp1, ixSetUpp, ZERO, tmp2);
                GetSubmatrix(dx, ixSetUpp, ZERO, dz2);
                DiagonalScale(LEFT, NORMAL, z2, dz2);
                dz2 += cU;
                DiagonalSolve(LEFT, NORMAL, tmp2, dz2); // dz1 = (bu-x)^-1 * (cU + z2*dx)
            }
                break;
            default:
                RuntimeError("Unrecognized method option.");
        }

        // dx, dy, dz1, dz2 should be computed at this point
        // Return stepx, stepz
        bool success = Linesearch(phi, mu, ACopy, bCopy, bl, bu, D1sq, D2sq, 
          x, y, z1, z2, r1, r2, center, Cinf0, cL, cU, stepx, stepz,
          dx, dy, dz1, dz2, ixSetLow, ixSetUpp, ixSetFix, dCol, beta, theta, ctrl);

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
            printf("%d :\t%1.1f\t%1.1f\t%1.1f\t%1.1f\t%1.1f\t%1.1f\t%1.1f\t%1.3f\t%1.3f\n", 
                numIts, Log(double(mu))/log10, Log(double(Pfeas))/log10,
                Log(double(Dfeas))/log10, Log(double(Cinf0))/log10, 
                Log(double(InfinityNorm(cL)))/log10, Log(double(InfinityNorm(cU)))/log10,
                Log(double(center))/log10, stepx, stepz);
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
                mu /= 0.5;
            }
        }

        // Update gradient and Hessian
        Copy(x, xin);
        xin *= beta;
        if( ctrl.outerEquil )
        {
            DiagonalSolve( LEFT, NORMAL, dCol, xin );
            if( ctrl.time )
                gradTimer.Start();
            phi.grad( xin, grad ); // get gradient
            if( ctrl.time )
            {
                gradTimer.Stop();
                hessTimer.Start();
            }
            phi.sparseHess( xin, Hess ); // get Hessian
            if( ctrl.time )
                hessTimer.Stop();
            DiagonalSolve( LEFT, NORMAL, dCol, grad );
            SymmetricDiagonalSolve( dCol, Hess );
        }
        else
        {
            if( ctrl.time )
                gradTimer.Start();
            phi.grad( xin, grad ); // get gradient
            if( ctrl.time )
            {
                gradTimer.Stop();
                hessTimer.Start();
            }
            phi.sparseHess( xin, Hess ); // get Hessian
            if( ctrl.time )
                hessTimer.Stop();
        }
        grad *= beta/theta;
        Hess *= beta*beta/theta;

        // Recompute residuals
        ResidualPD(ACopy, ixSetLow, ixSetUpp, ixSetFix,
          bCopy, D1sq, D2sq, grad, x, y, z1, z2, r1, r2);

        ResidualC(mu, ixSetLow, ixSetUpp, bl, bu, x, z1, z2, center, Cinf0, cL, cU);

        if (converged)
            break;
    } // end main for loop

    // Undo scaling by beta and zeta
    if( ctrl.scale )
    {
        x *= beta;
        y *= zeta;
        z1 *= zeta;
        z2 *= zeta;

        bl *= beta;
        bu *= beta;
    }

    // Undo scaling due to equilibration
    if( ctrl.outerEquil )
    {
        DiagonalSolve( LEFT, NORMAL, dCol, x );
        DiagonalSolve( LEFT, NORMAL, dRow, y );

        Matrix<Real> tmp;
        GetSubmatrix(dCol, ixSetLow, ZERO, tmp);
        DiagonalScale( LEFT, NORMAL, tmp, z1 );
        GetSubmatrix(dCol, ixSetUpp, ZERO, tmp);
        DiagonalScale( LEFT, NORMAL, tmp, z2 );

        DiagonalSolve( LEFT, NORMAL, dCol, bl );
        DiagonalSolve( LEFT, NORMAL, dCol, bu );
    }

    Int lowerActive;
    Int upperActive;
    // Report active constraints
    GetActiveConstraints(x, bl, bu, ixSetLow, ixSetUpp,
      lowerActive, upperActive);

    Output("========== Completed Newton's Method ==========");

    if( converged )
      Output("Result: Converged!");
    else
      Output("Result: Failed!");

    Copy(D2, D2sq);
    Copy(D1, D1sq);
    if( ctrl.outerEquil )
    {
        DiagonalScale(LEFT, NORMAL, dRow, D2sq);
        DiagonalScale(LEFT, NORMAL, dCol, D1sq);
    }
    DiagonalScale(LEFT, NORMAL, D2sq, D2sq);
    DiagonalScale(LEFT, NORMAL, D1sq, D1sq);

    // Reconstruct z from z1 and z2
    phi.grad( x, grad ); // get gradient
    ResidualPD(A, ixSetLow, ixSetUpp, IndexRange(0),
        b, D1sq, D2sq, grad, x, y, z1, z2, r1, r2);
    Matrix<Real> tmp;
    Zeros(z, n, 1);
    UpdateSubmatrix(z, ixSetLow, ZERO, Real(1), z1);
    UpdateSubmatrix(z, ixSetUpp, ZERO, Real(-1), z2);
    Matrix<Real> zFix;
    GetSubmatrix(r2, ixSetFix, ZERO, zFix);
    SetSubmatrix(z, ixSetFix, ZERO, zFix);

    // Compute unscaled residuals
    // TODO: Avoid recomputing some of this stuff
    ResidualPD(A, b, D1sq, D2sq, grad, x, y, z, r1, r2);
    ResidualC(mu, ixSetLow, ixSetUpp, bl, bu, x, z1, z2, center, Cinf0, cL, cU);

    Pfeas = InfinityNorm(r1);
    Dfeas = InfinityNorm(r2);
    Real cLInf = InfinityNorm(cL);
    Real cUInf = InfinityNorm(cU);
    Cfeas = Max(cLInf, cUInf);

    Real xInf = InfinityNorm(x);
    Real zInf = InfinityNorm(z);
    Output("  Pfeas         = ", Pfeas);
    Output("  Dfeas         = ", Dfeas);
    Output("  Cinf0         = ", Cinf0);
    Output("  |cL|oo        = ", cLInf);
    Output("  |cU|oo        = ", cUInf);
    Output("  Cinf0/|x||z|  = ", Cinf0/(xInf * zInf));
    Output("  |cL|oo/|x||z| = ", cLInf/(xInf * zInf));
    Output("  |cU|oo/|x||z| = ", cUInf/(xInf * zInf));
    Output("  center        = ", center);

    Output();
    Output("  Scaled:   max |x| = ", Max(x)/beta, "\tmax |y| = ", Max(y)/zeta, "\tmax |z| = ", Max(z)/zeta);
    Output("  Unscaled: max |x| = ", Max(x), "\tmax |y| = ", Max(y), "\tmax |z| = ", Max(z));
    Output();

    Output("  Number active constraints on");
    Output("    Lower bound: ", lowerActive);
    Output("    Upper bound: ", upperActive);

    if( ctrl.time )
    {
        totalTimer.Stop();

        Output("Timing results:");
        Output("  Total time:          ", totalTimer.Total());
        Output("    LDL time:          ", ldlTimer.Total());
        Output("    SolveAfter time:   ", solveAfterTimer.Total()); 
        Output("    Form KKT time:     ", kktTimer.Total());
        Output("    Form KKT rhs time: ", kktrhsTimer.Total());
        Output("    Hessian time:      ", hessTimer.Total());
        Output("    Grad time:         ", gradTimer.Total());
    }

}

#define PROTO(Real) \
  template void Newton \
  ( const pdco::PDCOObj<Real>& phi, \
    const Matrix<Real>& A, \
    const Matrix<Real>& b, \
          Matrix<Real>  bl, \
          Matrix<Real>  bu, \
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
          Matrix<Real>  bl, \
          Matrix<Real>  bu, \
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
