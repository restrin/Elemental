/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

#include "../util.hpp"

namespace El {
namespace pdco {

// Computes the residuals for the primal-dual equations
template<typename Real>
void ResidualPD
( const Matrix<Real>& A,
  const vector<Int>& ixSetLow,
  const vector<Int>& ixSetUpp,
  const vector<Int>& ixSetFix,
  const Matrix<Real>& b,
  const Matrix<Real>& D1sq,
  const Matrix<Real>& D2sq,
  const Matrix<Real>& grad,
  const Matrix<Real>& x,
  const Matrix<Real>& y,
  const Matrix<Real>& z1,
  const Matrix<Real>& z2,
  Matrix<Real>& r1,
  Matrix<Real>& r2 )
{
    EL_DEBUG_CSE

    vector<Int> ZERO (1,0);
    Int n = A.Width();

    Matrix<Real> tmp1;

    // Compute r1 = b - A*x - D2^2*y
    Copy(b, r1);
    Gemv(NORMAL, Real(-1), A, x, Real(1), r1); // r1 = b - A*x
    Hadamard(D2sq, y, tmp1); // tmp1 = D2^2 y
    r1 -= tmp1;

    // Compute r2 = grad + D1^2*x - A'*y - z1 + z2
    Copy(grad, r2);
    Hadamard(D1sq, x, tmp1); // tmp1 = D1^2*x
    r2 += tmp1; // r2 = grad + D1^2*x

    Gemv(TRANSPOSE, Real(-1), A, y, Real(1), r2); // r2 = grad + D1^2*x - A'*y
    UpdateSubmatrix(r2, ixSetLow, ZERO, Real(-1), z1); // r2 = grad + D1^2*x - A'*y - z1
    UpdateSubmatrix(r2, ixSetUpp, ZERO, Real(1), z2); // r2 = grad + D1^2*x - A'*y - z1 + z2

    // This is probably unnecessary
    Zeros(tmp1, ixSetFix.size(), 1);
    SetSubmatrix(r2, ixSetFix, ZERO, tmp1); // r2(fix) = 0
}

// Computes the residuals for the primal-dual equations
template<typename Real>
void ResidualPD
( const SparseMatrix<Real>& A,
  const vector<Int>& ixSetLow,
  const vector<Int>& ixSetUpp,
  const vector<Int>& ixSetFix,
  const Matrix<Real>& b,
  const Matrix<Real>& D1sq,
  const Matrix<Real>& D2sq,
  const Matrix<Real>& grad,
  const Matrix<Real>& x,
  const Matrix<Real>& y,
  const Matrix<Real>& z1,
  const Matrix<Real>& z2,
  Matrix<Real>& r1,
  Matrix<Real>& r2 )
{
    EL_DEBUG_CSE

    const vector<Int> ZERO (1,0);
    const Int n = A.Width();

    Matrix<Real> tmp1;

    // Compute r1 = b - A*x - D2^2*y
    Copy(b, r1);
    Multiply(NORMAL, Real(-1), A, x, Real(1), r1); // r1 = b - A*x
    Hadamard(D2sq, y, tmp1); // tmp1 = D2^2*y
    r1 -= tmp1;

    // Compute r2 = grad + D1^2*x - A'*y - z1 + z2
    Copy(grad, r2);
    Hadamard(D1sq, x, tmp1); // tmp2 = D1^2*x
    r2 += tmp1; // r2 = grad + D1^2*x

    Multiply(TRANSPOSE, Real(-1), A, y, Real(1), r2); // r2 = grad + D1^2*x - A'*y
    UpdateSubmatrix(r2, ixSetLow, ZERO, Real(-1), z1); // r2 = grad + D1^2*x - A'*y - z1
    UpdateSubmatrix(r2, ixSetUpp, ZERO, Real(1), z2); // r2 = grad + D1^2*x - A'*y - z1 + z2

    Zeros(tmp1, ixSetFix.size(), 1);
    SetSubmatrix(r2, ixSetFix, ZERO, tmp1); // r2(fix) = 0
}

// Compute the residuals for the complementarity conditions
template<typename Real>
void ResidualC
( const Real& mu,
  const vector<Int>& ixSetLow,
  const vector<Int>& ixSetUpp,
  const Matrix<Real>& bl,
  const Matrix<Real>& bu,
  const Matrix<Real>& x,
  const Matrix<Real>& z1,
  const Matrix<Real>& z2,
  Real& center,
  Real& Cinf0,
  Matrix<Real>& cL,
  Matrix<Real>& cU )
{
    EL_DEBUG_CSE
    
    const Real eps = limits::Epsilon<Real>();
    const vector<Int> ZERO (1,0);
    Matrix<Real> tmp1;
    Matrix<Real> tmp2;

    Real maxXz = 0;
    Real minXz = limits::Infinity<Real>();

    if( ixSetLow.size() > 0 )
    {
        // Compute cL = mu - (x - bl).*z1
        Ones(cL, ixSetLow.size(), 1);
        cL *= mu; // cL = mu
        Copy(x, tmp1);
        tmp1 -= bl;
        GetSubmatrix(tmp1, ixSetLow, ZERO, tmp2);
        DiagonalScale(LEFT, NORMAL, z1, tmp2); // (x - bl).*z1
        cL -= tmp2; // cL = mu - (x - bl).*z1

        maxXz = Max(maxXz, Max(tmp2));
        minXz = Min(minXz, Min(tmp2));
    }

    if( ixSetUpp.size() > 0 )
    {
        // Compute cU = mu - (bu - x).*z2
        Ones(cU, ixSetUpp.size(), 1);
        cU *= mu; // cU = mu
        Copy(bu, tmp1);
        tmp1 -= x;
        GetSubmatrix(tmp1, ixSetUpp, ZERO, tmp2);
        DiagonalScale(LEFT, NORMAL, z2, tmp2); // (bu - x).*z2
        cU -= tmp2; // cU = mu - (bu - x).*z2
        
        maxXz = Max(maxXz, Max(tmp2));
        minXz = Min(minXz, Min(tmp2));
    }

    if( ixSetLow.size() == 0 && ixSetUpp.size() == 0 )
    {
        center = Real(1.0);
        Cinf0 = Real(0);
    }
    else
    {
        // Keep things safe against division by 0 or Nan
        maxXz = Max(maxXz, eps);
        minXz = Max(minXz, eps);
        center = maxXz / minXz;
        Cinf0 = maxXz;
    }
}

#define PROTO(Real) \
  template void ResidualPD \
  ( const Matrix<Real>& A, \
    const vector<Int>& ixSetLow, \
    const vector<Int>& ixSetUpp, \
    const vector<Int>& ixSetFix, \
    const Matrix<Real>& b, \
    const Matrix<Real>& D1sq, \
    const Matrix<Real>& D2sq, \
    const Matrix<Real>& grad, \
    const Matrix<Real>& x, \
    const Matrix<Real>& y, \
    const Matrix<Real>& z1, \
    const Matrix<Real>& z2, \
    Matrix<Real>& r1, \
    Matrix<Real>& r2 ); \
  template void ResidualPD \
  ( const SparseMatrix<Real>& A, \
    const vector<Int>& ixSetLow, \
    const vector<Int>& ixSetUpp, \
    const vector<Int>& ixSetFix, \
    const Matrix<Real>& b, \
    const Matrix<Real>& D1sq, \
    const Matrix<Real>& D2sq, \
    const Matrix<Real>& grad, \
    const Matrix<Real>& x, \
    const Matrix<Real>& y, \
    const Matrix<Real>& z1, \
    const Matrix<Real>& z2, \
    Matrix<Real>& r1, \
    Matrix<Real>& r2 ); \
  template void ResidualC \
  ( const Real& mu, \
    const vector<Int>& ixSetLow, \
    const vector<Int>& ixSetUpp, \
    const Matrix<Real>& bl, \
    const Matrix<Real>& bu, \
    const Matrix<Real>& x, \
    const Matrix<Real>& z1, \
    const Matrix<Real>& z2, \
    Real& center, \
    Real& Cinf0, \
    Matrix<Real>& cL, \
    Matrix<Real>& cU );

#define EL_NO_INT_PROTO
#define EL_NO_COMPLEX_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include "El/macros/Instantiate.h"

} // namespace pdco
} // namespace El
