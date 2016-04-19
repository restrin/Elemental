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

template<typename Real>
void FormHandW
( const Matrix<Real>& Hess,
  const Matrix<Real>& D1,
  const Matrix<Real>& x,
  const Matrix<Real>& z1,
  const Matrix<Real>& z2,
  const Matrix<Real>& bl,
  const Matrix<Real>& bu,
  const vector<Int>& ixSetLow,
  const vector<Int>& ixSetUpp,
  const vector<Int>& ixSetFix,
  const Matrix<Real>& r2,
  const Matrix<Real>& cL,
  const Matrix<Real>& cU,
        Matrix<Real>& H,
        Matrix<Real>& w )
{
    DEBUG_ONLY(CSE cse("pdco::FormHandW"))

    vector<Int> ZERO (1,0);
    Matrix<Real> tmp1;
    Matrix<Real> tmp2;
    Matrix<Real> tmp3;

    // Form H = Hess + D1^2 + (x-bl)^-1*z1 - (bu-x)^-1*z2
    Copy(Hess, H);
    Copy(D1, tmp1);
    DiagonalScale(LEFT, NORMAL, D1, tmp1);
    UpdateDiagonal(H, Real(1), tmp1, 0); // H = Hess + D1^2
    
    // Form w = r2 - (x-bl)^-1*cL + (bu-x)^-1*cU
    Copy(r2, w);

    // Form (x-bl)^-1*z1
    Copy(x, tmp1);
    tmp1 -= bl;
    GetSubmatrix(tmp1, ixSetLow, ixSetLow, tmp2);
    Copy(z1, tmp1);
    DiagonalSolve(LEFT, NORMAL, tmp2, tmp1, false);
    // H = Hess + D1^2 + (x-bl)^-1*z1
    UpdateSubmatrix(H, ixSetLow, ixSetLow, Real(1), tmp1);

    // Form (x-bl)^-1*cL
    Copy(cL, tmp3);
    DiagonalSolve(LEFT, NORMAL, tmp2, tmp3);
    // w = r2 - (x-bl)^-1*cL
    UpdateSubmatrix(w, ixSetLow, ZERO, Real(-1), tmp3);

    // Form (bu-x)^-1*z2
    Copy(bu, tmp1);
    tmp1 -= x;
    GetSubmatrix(tmp1, ixSetUpp, ixSetUpp, tmp2);
    Copy(z2, tmp1);
    DiagonalSolve(LEFT, NORMAL, tmp2, tmp1, false);
    // H = Hess + D1^2 + (x-bl)^-1*z1 - (bu-x)^-1*z2
    UpdateSubmatrix(H, ixSetUpp, ixSetUpp, Real(-1), tmp1);

    // Form (x-bu)^-1*cU
    Copy(cU, tmp3);
    DiagonalSolve(LEFT, NORMAL, tmp2, tmp3);
    // w = r2 - (x-bl)^-1*cL + (bu-x)^-1*cU
    UpdateSubmatrix(w, ixSetUpp, ZERO, Real(1), tmp3);
}

#define PROTO(Real) \
  template void FormHandW \
  ( const Matrix<Real>& Hess, \
    const Matrix<Real>& D1, \
    const Matrix<Real>& x, \
    const Matrix<Real>& z1, \
    const Matrix<Real>& z2, \
    const Matrix<Real>& bl, \
    const Matrix<Real>& bu, \
    const vector<Int>& ixSetLow, \
    const vector<Int>& ixSetUpp, \
    const vector<Int>& ixSetFix, \
    const Matrix<Real>& r2, \
    const Matrix<Real>& cL, \
    const Matrix<Real>& cU, \
          Matrix<Real>& H, \
          Matrix<Real>& w );

#define EL_NO_INT_PROTO
#define EL_NO_COMPLEX_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include "El/macros/Instantiate.h"

} // namespace pdco
} // namespace El
