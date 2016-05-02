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
void ClassifyBounds
( const Matrix<Real>& bl,
  const Matrix<Real>& bu,
        vector<Int>& ixSetLow,
        vector<Int>& ixSetUpp,
        vector<Int>& ixSetFix,
  bool print );

template<typename Real>
void Initialize
(       Matrix<Real>& x,
        Matrix<Real>& y,
        Matrix<Real>& z1,
        Matrix<Real>& z2,
  const Matrix<Real>& bl,
  const Matrix<Real>& bu,
  const vector<Int>& ixSetLow,
  const vector<Int>& ixSetUpp,
  const vector<Int>& ixSetFix,
  const Real& x0min,
  const Real& z0min,
  const Int& m,
  const Int& n,
  bool print );

template<typename Real>
void ResidualPD
( const Matrix<Real>& A,
  const vector<Int>& ixSetLow,
  const vector<Int>& ixSetUpp,
  const vector<Int>& ixSetFix,
  const Matrix<Real>& b,
  const Matrix<Real>& D1,
  const Matrix<Real>& D2,
  const Matrix<Real>& grad,
  const Matrix<Real>& x,
  const Matrix<Real>& y,
  const Matrix<Real>& z1,
  const Matrix<Real>& z2,
  Matrix<Real>& r1,
  Matrix<Real>& r2 );

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
  Real& Cfeas,
  Matrix<Real>& cL,
  Matrix<Real>& cU );

template<typename Real>
bool Linesearch
( const PDCOObj<Real>& phi,
  const Real& mu,
  const Matrix<Real>& A,
  const Matrix<Real>& b, 
  const Matrix<Real>& bl,
  const Matrix<Real>& bu,
  const Matrix<Real>& D1,
  const Matrix<Real>& D2,
        Matrix<Real>& x,
        Matrix<Real>& y,
        Matrix<Real>& z1,
        Matrix<Real>& z2,
        Matrix<Real>& r1,
        Matrix<Real>& r2,
        Real& center,
        Real& Cinf0,
        Matrix<Real>& cL,
        Matrix<Real>& cU,
        Real& stepx,
        Real& stepz,
  const Matrix<Real>& dx,
  const Matrix<Real>& dy,
  const Matrix<Real>& dz1,
  const Matrix<Real>& dz2,
  const vector<Int>& ixSetLow,
  const vector<Int>& ixSetUpp,
  const vector<Int>& ixSetFix,
  const PDCOCtrl<Real>& ctrl );

template<typename Real>
Real Merit
( const Matrix<Real>& r1,
  const Matrix<Real>& r2,
  const Matrix<Real>& cL,
  const Matrix<Real>& cU );

template<typename Real>
Real MaxXStepSize
( const Matrix<Real>& x,
  const Matrix<Real>& dx,
  const Matrix<Real>& bl,
  const Matrix<Real>& bu,
  const vector<Int>& ixSetLow,
  const vector<Int>& ixSetUpp );

template<typename Real>
Real MaxZStepSize
( const Matrix<Real>& z,
  const Matrix<Real>& dz );

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
        Matrix<Real>& w,
  const bool diagHess );

template <typename Real>
void UpdateSubdiagonal
( Matrix<Real>& A,
  const vector<Int>& ixSet,
  const Real& alpha,
  Matrix<Real>& dSub );

vector<Int> IndexRange(Int n);

template <typename Real>
void GetActiveConstraints
( const Matrix<Real>& x,
  const Matrix<Real>& bl,
  const Matrix<Real>& bu,
  const vector<Int>& ixSetLow,
  const vector<Int>& ixSetUpp,
        Int& lowerActive,
        Int& upperActive );

} // namespace pdco
} // namespace El
