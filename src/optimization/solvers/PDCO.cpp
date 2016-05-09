/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>
#include "./PDCO/IPM.hpp"

namespace El {

template<typename Real>
void PDCO
( const pdco::PDCOObj<Real>& phi,
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
  const pdco::Ctrl<Real>& ctrl )
{
    DEBUG_ONLY(CSE cse("PDCO"))

    pdco::Newton(phi, A, b, bl, bu, D1, D2, x, r, y, z, ctrl.pdcoCtrl);
}

template<typename Real>
void PDCO
( const pdco::PDCOObj<Real>& phi,
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
  const pdco::Ctrl<Real>& ctrl )
{
    DEBUG_ONLY(CSE cse("PDCO"))

    pdco::Newton(phi, A, b, bl, bu, D1, D2, x, r, y, z, ctrl.pdcoCtrl);
}

#define PROTO(Real) \
  template void PDCO \
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
    const pdco::Ctrl<Real>& ctrl ); \
  template void PDCO \
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
    const pdco::Ctrl<Real>& ctrl );

#define EL_NO_INT_PROTO
#define EL_NO_COMPLEX_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El
