/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

namespace El {

template<typename Real>
void LowerClip( Matrix<Real>& X, const Real& lowerBound )
{
    EL_DEBUG_CSE
    auto lowerClip = [&]( const Real& alpha ) { return Max(lowerBound,alpha); };
    EntrywiseMap( X, MakeFunction(lowerClip) );
}

template<typename Real>
void UpperClip( Matrix<Real>& X, const Real& upperBound )
{
    EL_DEBUG_CSE
    auto upperClip = [&]( const Real& alpha ) { return Min(upperBound,alpha); };
    EntrywiseMap( X, MakeFunction(upperClip) );
}

template<typename Real>
void Clip( Matrix<Real>& X, const Real& lowerBound, const Real& upperBound )
{
    EL_DEBUG_CSE
    auto clip = [&]( const Real& alpha )
      { return Max(lowerBound,Min(upperBound,alpha)); };
    EntrywiseMap( X, MakeFunction(clip) );
}

template<typename Real>
void LowerClip( AbstractDistMatrix<Real>& X, const Real& lowerBound )
{ LowerClip( X.Matrix(), lowerBound ); }

template<typename Real>
void UpperClip( AbstractDistMatrix<Real>& X, const Real& upperBound )
{ UpperClip( X.Matrix(), upperBound ); }

template<typename Real>
void Clip
( AbstractDistMatrix<Real>& X, const Real& lowerBound, const Real& upperBound )
{ Clip( X.Matrix(), lowerBound, upperBound ); }

template<typename Real>
void LowerClip( DistMultiVec<Real>& X, const Real& lowerBound )
{ LowerClip( X.Matrix(), lowerBound ); }

template<typename Real>
void UpperClip( DistMultiVec<Real>& X, const Real& upperBound )
{ UpperClip( X.Matrix(), upperBound ); }

template<typename Real>
void Clip
( DistMultiVec<Real>& X, const Real& lowerBound, const Real& upperBound )
{ Clip( X.Matrix(), lowerBound, upperBound ); }

#define PROTO(Real) \
  template void LowerClip \
  ( Matrix<Real>& X, const Real& lowerBound ); \
  template void LowerClip \
  ( AbstractDistMatrix<Real>& X, const Real& lowerBound ); \
  template void LowerClip \
  ( DistMultiVec<Real>& X, const Real& lowerBound ); \
  template void UpperClip \
  ( Matrix<Real>& X, const Real& upperBound ); \
  template void UpperClip \
  ( AbstractDistMatrix<Real>& X, const Real& upperBound ); \
  template void UpperClip \
  ( DistMultiVec<Real>& X, const Real& upperBound ); \
  template void Clip \
  ( Matrix<Real>& X, const Real& lowerBound, const Real& upperBound ); \
  template void Clip \
  ( AbstractDistMatrix<Real>& X, \
    const Real& lowerBound, const Real& upperBound ); \
  template void Clip \
  ( DistMultiVec<Real>& X, const Real& lowerBound, const Real& upperBound );

#define EL_NO_INT_PROTO
#define EL_NO_COMPLEX_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El
