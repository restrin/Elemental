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

vector<Int> IndexRange(Int n)
{
  vector<Int> range (n,0);
  for (Int i = 0; i < n; i++)
  {
    range[i] = i;
  }
  return range;
}

// Updates a (possibly non-) contiguous part
// of the diagonal.
template<typename Real>
void UpdateSubdiagonal
( Matrix<Real>& A,
  const vector<Int>& ixSet,
  const Real& alpha,
  const Matrix<Real>& dSub )
{
    const Real* dbuf = dSub.LockedBuffer();
          Real* Abuf = A.Buffer();
    const Int ALDim = A.LDim();

    for( Int i=0; i < ixSet.size(); i++ )
    {
        Int ix = ixSet[i];
        Abuf[ix + ix*ALDim] += alpha*dbuf[i];
    }
}

// Queues update for a (possibly non-) contiguous part
// of the diagonal of a sparse matrix
// A call to A.ProcessQueues() is expected afterwards
template<typename Real>
void QueueUpdateSubdiagonal
( SparseMatrix<Real>& A,
  const vector<Int>& ixSet,
  const Real& alpha,
  const Matrix<Real>& dSub )
{
    const Real* dbuf = dSub.LockedBuffer();
    for( Int i=0; i < ixSet.size(); i++ )
    {
        A.QueueUpdate(ixSet[i], ixSet[i], alpha*dbuf[i]);
    }
}

// Get the number of active bound constraints
template<typename Real>
void GetActiveConstraints
( const Matrix<Real>& x,
  const Matrix<Real>& bl,
  const Matrix<Real>& bu,
  const vector<Int>& ixSetLow,
  const vector<Int>& ixSetUpp,
        Int& lowerActive,
        Int& upperActive )
{
    Int n = x.Height();
    lowerActive = 0;
    upperActive = 0;
    Int ctrLow = 0;
    Int ctrUpp = 0;
    for( Int i = 0; i < n; i++ )
    {
        if( i == ixSetLow[ctrLow])
        {
            if( x.Get(i,0) - bl.Get(i,0) < 1e-8 )
                lowerActive++;
            ctrLow++;
        }
        if( i == ixSetUpp[ctrUpp])
        {
            if( bu.Get(i,0) - x.Get(i,0) < 1e-8 )
                upperActive++;
            ctrUpp++;
        }
    }
}

#define PROTO(Real) \
  vector<Int> IndexRange(Int n); \
  template void UpdateSubdiagonal \
  ( Matrix<Real>& A, \
    const vector<Int>& ixSet, \
    const Real& alpha, \
    const Matrix<Real>& dSub ); \
  template void QueueUpdateSubdiagonal \
  ( SparseMatrix<Real>& A, \
    const vector<Int>& ixSet, \
    const Real& alpha, \
    const Matrix<Real>& dSub ); \
  template void GetActiveConstraints \
  ( const Matrix<Real>& x, \
    const Matrix<Real>& bl, \
    const Matrix<Real>& bu, \
    const vector<Int>& ixSetLow, \
    const vector<Int>& ixSetUpp, \
          Int& lowerActive, \
          Int& upperActive );



#define EL_NO_INT_PROTO
#define EL_NO_COMPLEX_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include "El/macros/Instantiate.h"

} // namespace pdco
} // namespace El
