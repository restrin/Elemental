/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"
#include "./util.hpp"

namespace El {
namespace pdco {

// The following solves the following convex problem:
//
//   minimize phi(x) + 1/2 ||D1*x||^2 + 1/2 ||r||^2
//     x,r
//   s.t.     A*x + D2*r = b, bl <= x <= bu, r unconstrained
//
// using Newton's method.
//

template<typename Real>
void Newton
( const pdco::PDCOObj<Real>& phi,
  const Matrix<Real>& A,
  const Matrix<Real>& b, 
  const Matrix<Real>& bl,
  const Matrix<Real>& bu,
  const Matrix<Real>& D1,
  const Matrix<Real>& D2,
        Matrix<Real>& x, 
        Matrix<Real>& y,
        Matrix<Real>& z,
  const PDCOCtrl<Real>& ctrl )
{
    DEBUG_ONLY(CSE cse("pdco::Newton"))    
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
