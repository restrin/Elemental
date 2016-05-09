/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

namespace El {
namespace pdco {

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
        Matrix<Real>& r,
        Matrix<Real>& y,
        Matrix<Real>& z,
  const PDCOCtrl<Real>& ctrl=PDCOCtrl<Real>() );

template<typename Real>
void Newton
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
  const PDCOCtrl<Real>& ctrl=PDCOCtrl<Real>() );

} // namespace pdco
} // namespace El
