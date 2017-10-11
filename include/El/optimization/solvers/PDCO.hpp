/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   Copyright (c) 2016, Ron Estrin
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_OPTIMIZATION_SOLVERS_PDCO_HPP
#define EL_OPTIMIZATION_SOLVERS_PDCO_HPP

#include <El/optimization/solvers/util.hpp>

namespace El {

namespace pdco
{
enum Method {
  LDLy, // Solve (A D1^2 A' + D2^2 I) for dy via sparse LDL
  LDLx, // Solve (D1 A' A D1 + D2^2 I) for dx via sparse LDL
  LDL2, // Solve 2x2 KKT system with sparse LDL
  LDL25 // Solve Diagonally scaled 2x2 KKT system with sparse LDL
};
} // namespace pdco

template<typename Real>
struct PDCOCtrl
{
    // Feasibility tolerance
    // Accuracy for satisfying linear constraints
    Real feaTol=Pow(limits::Epsilon<Real>(),Real(0.5));

    // Optimality tolerance
    Real optTol=Pow(limits::Epsilon<Real>(),Real(0.5));

    // Min distance between x0 and bl or bu AFTER SCALING.
    Real x0min=Real(1.0);

    // Min distance between x0 and bl or bu AFTER SCALING.
    Real z0min=Real(1.0);

    // Maximum iterations of primal-dual barrier method
    int maxIts=100;

    // Print progress of the Interior point method?
    bool print=false;

    // What method to use to get search directions
    pdco::Method method=pdco::LDLy;

    // Initial mu, should be >= 0
    Real mu0 = Real(0.1);

    // Keep step sizes for x,z the same?
    bool stepSame = true;

    // Maximum number of backtracking linesearch iterations
    Int maxLSIts = 10;

    // Linesearch backtracking parameter
    Real lsparam = Real(0.5);

    // Controls how close x or z can be to their respective bounds
    // Must be between (0,1)
    Real stepTol = Real(0.99);

    // Sufficient descent tolerance
    Real eta = Real(1e-4);

    // mu is reduced if center < bigcenter
    Real bigcenter = Real(1e3);

    // Temporary regularization level (for sparse LDL)
    Real deltaTmp = Pow(limits::Epsilon<Real>(),Real(0.25));

    // Use backtracking linesearch?
    bool backtrack = true;

    // Update mu ala the original PDCO (vs. naive)?
    bool adaptiveMu = true;

    // Perform geometric-mean equilibration?
    bool outerEquil = true;

    // Scale the input data
    // Typically results in better convergence
    bool scale = true;

    // The controls for quasi-(semi)definite solves
    RegSolveCtrl<Real> solveCtrl;

    // Profile timings?
    bool time = false;
};

namespace pdco
{
// The following solves the following convex problem:
//
//   minimize phi(x) + 1/2 ||D1*x||^2 + 1/2 ||r||^2
//     x,r
//   s.t.     A*x + D2*r = b, bl <= x <= bu, r unconstrained
//
// using Newton's method.
//

// Control structure for the high-level pdco solver
// ------------------------------------------------------------------
template<typename Real>
struct Ctrl
{
    PDCOCtrl<Real> pdcoCtrl;

    Ctrl()
    {

    }
};

// Structure representing objective function for pdco solver
template<typename Real>
class PDCOObj
{
    public:
        // Function handles for computing objective, gradient and Hessian.
        // First argument is primal variable x,
        // Second argument is output.
        void (*objLocal)(Matrix<Real>&, Real&) = 0; // Objective value
        void (*gradLocal)(Matrix<Real>&, Matrix<Real>&) = 0; // Gradient
        void (*hessLocal)(Matrix<Real>&, Matrix<Real>&) = 0; // Hessian
        void (*sparseHessLocal)(Matrix<Real>&, SparseMatrix<Real>&) = 0; // Sparse Hessian

        Matrix<Real> scale;
        Real beta = 1;
        Real theta = 1;
        Real zeta = 1;

        Matrix<Real> xin;

    public:
        PDCOObj(Int n)
        {
            Zeros( this->xin, n, 1 );
            Ones( this->scale, n, 1 );
        }

        void obj(Matrix<Real>& x, Real& val)
        {
            Real beta = this->beta;
            Copy(x, this->xin);
            this->xin *= beta;

            DiagonalSolve( LEFT, NORMAL, this->scale, this->xin );
            this->objLocal( this->xin, val );
        }

        void grad(Matrix<Real>& x, Matrix<Real>& g)
        {
            Real beta = this->beta;
            Real theta = this->theta;
            Copy(x, this->xin);
            this->xin *= beta;

            DiagonalSolve( LEFT, NORMAL, this->scale, this->xin );
            this->gradLocal( this->xin, g );
            DiagonalSolve( LEFT, NORMAL, this->scale, g );
            g *= beta/theta;
        }

        void hess(Matrix<Real>& x, Matrix<Real>& H)
        {
            Real beta = this->beta;
            Real theta = this->theta;
            Copy(x, this->xin);
            this->xin *= beta;
            
            DiagonalSolve( LEFT, NORMAL, this->scale, this->xin ); 
            this->hessLocal( this->xin, H );
            SymmetricDiagonalSolve( this->scale, H);
            H *= beta*beta/theta;
        }

        void sparseHess(Matrix<Real>& x, SparseMatrix<Real>& H)
        {
            Real beta = this->beta;
            Real theta = this->theta;
            Copy(x, this->xin);
            this->xin *= beta;
            
            DiagonalSolve( LEFT, NORMAL, this->scale, this->xin ); 
            this->sparseHessLocal( this->xin, H );
            SymmetricDiagonalSolve( this->scale, H);
            H *= beta*beta/theta;
        }
};

} // namespace pdco 

template<typename Real>
void PDCO
(       pdco::PDCOObj<Real>& phi,
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
  const pdco::Ctrl<Real>& ctrl=pdco::Ctrl<Real>() );

template<typename Real>
void PDCO
(       pdco::PDCOObj<Real>& phi,
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
  const pdco::Ctrl<Real>& ctrl=pdco::Ctrl<Real>() );

} // namespace El

#endif // ifndef EL_OPTIMIZATION_SOLVERS_PDCO_HPP