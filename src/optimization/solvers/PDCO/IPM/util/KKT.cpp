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
        Matrix<Real>& w,
  const bool diagHess )
{
    DEBUG_ONLY(CSE cse("pdco::FormHandW"))

    const vector<Int> ZERO (1,0);
    Matrix<Real> tmp1;
    Matrix<Real> tmp2;
    Matrix<Real> tmp3;

    // Form H = Hess + D1^2 + (x-bl)^-1*z1 + (bu-x)^-1*z2
    Copy(Hess, H);
    Copy(D1, tmp1);
    DiagonalScale(LEFT, NORMAL, D1, tmp1);
    if( diagHess )
      Axpy(Real(1), tmp1, H);
    else
      UpdateDiagonal(H, Real(1), tmp1, 0); // H = Hess + D1^2

    // Form w = -r2 + (x-bl)^-1*cL - (bu-x)^-1*cU
    Copy(r2, w);
    w *= -1;

    // Form (x-bl)^-1*z1
    Copy(x, tmp1);
    tmp1 -= bl;
    GetSubmatrix(tmp1, ixSetLow, ZERO, tmp2);
    Copy(z1, tmp1);
    DiagonalSolve(LEFT, NORMAL, tmp2, tmp1, false);
    // H = Hess + D1^2 + (x-bl)^-1*z1
    if( diagHess )
      UpdateSubmatrix(H, ixSetLow, ZERO, Real(1), tmp1);
    else
      UpdateSubdiagonal(H, ixSetLow, Real(1), tmp1);

    // Form (x-bl)^-1*cL
    Copy(cL, tmp3);
    DiagonalSolve(LEFT, NORMAL, tmp2, tmp3);
    // w = -r2 + (x-bl)^-1*cL
    UpdateSubmatrix(w, ixSetLow, ZERO, Real(1), tmp3);

    // Form (bu-x)^-1*z2
    Copy(bu, tmp1);
    tmp1 -= x;
    GetSubmatrix(tmp1, ixSetUpp, ZERO, tmp2);
    Copy(z2, tmp1);
    DiagonalSolve(LEFT, NORMAL, tmp2, tmp1, false);
    // H = Hess + D1^2 + (x-bl)^-1*z1 + (bu-x)^-1*z2
    if( diagHess )
      UpdateSubmatrix(H, ixSetUpp, ZERO, Real(1), tmp1);
    else
      UpdateSubdiagonal(H, ixSetUpp, Real(1), tmp1);

    // Form (bu-x)^-1*cU
    Copy(cU, tmp3);
    DiagonalSolve(LEFT, NORMAL, tmp2, tmp3);
    // w = -r2 + (x-bl)^-1*cL - (bu-x)^-1*cU
    UpdateSubmatrix(w, ixSetUpp, ZERO, Real(-1), tmp3);
}

template<typename Real>
void FormKKT
(       SparseMatrix<Real>& Hess,
  const SparseMatrix<Real>& A,
  const Matrix<Real>& D1sq,
  const Matrix<Real>& D2sq,
  const Matrix<Real>& x,
  const Matrix<Real>& z1,
  const Matrix<Real>& z2,
  const Matrix<Real>& bl,
  const Matrix<Real>& bu,
  const vector<Int>& ixSetLow,
  const vector<Int>& ixSetUpp,
  const vector<Int>& ixSetFix,
        SparseMatrix<Real>& K )
{
    DEBUG_ONLY(CSE cse("pdco::FormKKT"))

    const Int m = A.Height();
    const Int n = A.Width();
    const vector<Int> ALL_n = IndexRange(n);
    const vector<Int> ZERO (1,0);

    Matrix<Real> tmp1;
    Matrix<Real> tmp2;
    
    QueueUpdateSubdiagonal(Hess, ALL_n, Real(1), D1sq);

    // Form (x-bl)^-1*z1
    Copy(x, tmp1);
    tmp1 -= bl;
    GetSubmatrix(tmp1, ixSetLow, ZERO, tmp2);
    Copy(z1, tmp1);
    DiagonalSolve(LEFT, NORMAL, tmp2, tmp1, false);
    // Queue update for H + (x-bl)\z1
    QueueUpdateSubdiagonal(Hess, ixSetLow, Real(1), tmp1);

    // Form (bu-x)^-1*z2
    Copy(bu, tmp1);
    tmp1 -= x;
    GetSubmatrix(tmp1, ixSetUpp, ZERO, tmp2);
    Copy(z2, tmp1);
    DiagonalSolve(LEFT, NORMAL, tmp2, tmp1, false);
    // Queue update for H + (x-bl)\z1 + (bu-x)\z2
    QueueUpdateSubdiagonal(Hess, ixSetUpp, Real(1), tmp1);

    Hess.ProcessQueues();

    if( ixSetFix.size() > 0 )
    {
        // Fix the Hessian to account for fixed entries
    }

    const Int numEntriesH = Hess.NumEntries();
    const Int numEntriesA = A.NumEntries();

    Zeros( K, m+n, m+n );
    K.Reserve( 2*numEntriesA + numEntriesH + m);

    // Set K(1:n,1:n) = Hess
    for( Int e=0; e<numEntriesH; ++e )
        K.QueueUpdate( Hess.Row(e), Hess.Col(e), Hess.Value(e) );

    // Set K(n+1:n+m,1:n) = A and K(1:n,n+1:n+m) = A'
    for( Int e=0; e<numEntriesA; ++e )
    {
        K.QueueUpdate( A.Row(e)+n, A.Col(e), A.Value(e) );
        K.QueueUpdate( A.Col(e), A.Row(e)+n, A.Value(e) );
    }

    // Set K(n+1:n+m,n+1:n+m)=-D2^2
    const Real* dbuf = D2sq.LockedBuffer();
    for( Int i=0; i<m; i++ )
        K.QueueUpdate( n+i, n+i, -dbuf[i]);

    K.ProcessQueues();
    K.FreezeSparsity();
}

template<typename Real>
void FormKKTRHS
( const Matrix<Real>& x,
  const Matrix<Real>& r1,
  const Matrix<Real>& r2,
  const Matrix<Real>& cL,
  const Matrix<Real>& cU,
  const Matrix<Real>& bl,
  const Matrix<Real>& bu,
  const vector<Int>& ixSetLow,
  const vector<Int>& ixSetUpp,
        Matrix<Real>& w )
{
    DEBUG_ONLY(CSE cse("pdco::FormKKTRHS"))

    const vector<Int> ZERO (1,0);
    const Int m = r1.Height();
    const Int n = r2.Height();

    Matrix<Real> tmp1;
    Matrix<Real> tmp2;

    Zeros(w, n+m, 1);
    // Form w = [w1] = [ -r2 + (x-bl)\cL - (bu-x)\cU ]
    //          [w2] = [ r1                          ]

    auto w1 = w(IR(0,n), IR(0));
    Copy(r2, w1);
    w1 *= -1;
    auto w2 = w(IR(n,END), IR(0));
    Copy(r1, w2);

    // Form (x-bl)^-1*z1
    Copy(x, tmp1);
    tmp1 -= bl;
    GetSubmatrix(tmp1, ixSetLow, ZERO, tmp2);
    // Form (x-bl)^-1*cL
    Copy(cL, tmp1);
    DiagonalSolve(LEFT, NORMAL, tmp2, tmp1);
    // w = -r2 + (x-bl)^-1*cL
    UpdateSubmatrix(w, ixSetLow, ZERO, Real(1), tmp1);

    // Form (bu-x)^-1*z2
    Copy(bu, tmp1);
    tmp1 -= x;
    GetSubmatrix(tmp1, ixSetUpp, ZERO, tmp2);
    // Form (bu-x)^-1*cU
    Copy(cU, tmp1);
    DiagonalSolve(LEFT, NORMAL, tmp2, tmp1);
    // w = -r2 + (x-bl)^-1*cL - (bu-x)^-1*cU
    UpdateSubmatrix(w, ixSetUpp, ZERO, Real(-1), tmp1);
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
          Matrix<Real>& w, \
    const bool diagHess ); \
  template void FormKKT \
  (       SparseMatrix<Real>& Hess, \
    const SparseMatrix<Real>& A, \
    const Matrix<Real>& D1sq, \
    const Matrix<Real>& D2sq, \
    const Matrix<Real>& x, \
    const Matrix<Real>& z1, \
    const Matrix<Real>& z2, \
    const Matrix<Real>& bl, \
    const Matrix<Real>& bu, \
    const vector<Int>& ixSetLow, \
    const vector<Int>& ixSetUpp, \
    const vector<Int>& ixSetFix, \
          SparseMatrix<Real>& K ); \
  template void FormKKTRHS \
  ( const Matrix<Real>& x, \
    const Matrix<Real>& r1, \
    const Matrix<Real>& r2, \
    const Matrix<Real>& z1, \
    const Matrix<Real>& z2, \
    const Matrix<Real>& bl, \
    const Matrix<Real>& bu, \
    const vector<Int>& ixSetLow, \
    const vector<Int>& ixSetUpp, \
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
