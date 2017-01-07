/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include <El/blas_like/level1.hpp>

namespace El {

template<typename Real,
         typename/*=EnableIf<IsReal<Real>>*/>
ValueInt<Real> VectorMinLoc( const Matrix<Real>& x )
{
    EL_DEBUG_CSE
    const Int m = x.Height();
    const Int n = x.Width();
    EL_DEBUG_ONLY(
      if( m != 1 && n != 1 )
          LogicError("Input should have been a vector");
    )
    ValueInt<Real> pivot;
    pivot.index = -1;
    pivot.value = limits::Max<Real>();
    if( n == 1 )
    {
        for( Int i=0; i<m; ++i )
        {
            const Real value = x.Get(i,0);
            if( value < pivot.value )
            {
                pivot.value = value;
                pivot.index = i;
            }
        }
    }
    else
    {
        for( Int j=0; j<n; ++j )
        {
            const Real value = x.Get(0,j);
            if( value < pivot.value )
            {
                pivot.value = value;
                pivot.index = j;
            }
        }
    }
    return pivot;
}

template<typename Real,
         typename/*=EnableIf<IsReal<Real>>*/>
ValueInt<Real> VectorMinLoc( const AbstractDistMatrix<Real>& x )
{
    EL_DEBUG_CSE
    const Int m = x.Height();
    const Int n = x.Width();
    EL_DEBUG_ONLY(
      if( m != 1 && n != 1 )
          LogicError("Input should have been a vector");
      if( !x.Grid().InGrid() )
          LogicError("viewing processes are not allowed");
    )
    ValueInt<Real> pivot;
    pivot.index = -1;
    pivot.value = limits::Max<Real>();
    if( x.Participating() )
    {
        if( n == 1 )
        {
            if( x.RowRank() == x.RowAlign() )
            {
                const Int mLocal = x.LocalHeight();
                for( Int iLoc=0; iLoc<mLocal; ++iLoc )
                {
                    const Real value = x.GetLocal(iLoc,0);
                    if( value < pivot.value )
                    {
                        pivot.value = value;
                        pivot.index = x.GlobalRow(iLoc);
                    }
                }
            }
        }
        else
        {
            if( x.ColRank() == x.ColAlign() )
            {
                const Int nLocal = x.LocalWidth();
                for( Int jLoc=0; jLoc<nLocal; ++jLoc )
                {
                    const Real value = x.GetLocal(0,jLoc);
                    if( value < pivot.value )
                    {
                        pivot.value = value;
                        pivot.index = x.GlobalCol(jLoc);
                    }
                }
            }
        }
        pivot = mpi::AllReduce( pivot, mpi::MinLocOp<Real>(), x.DistComm() );
    }
    mpi::Broadcast( pivot, x.Root(), x.CrossComm() );
    return pivot;
}

template<typename Real,
         typename/*=EnableIf<IsReal<Real>>*/>
ValueInt<Real> VectorMinLoc( const DistMultiVec<Real>& x )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if( x.Width() != 1 )
          LogicError("Input should have been a vector");
    )
    ValueInt<Real> pivot;
    pivot.index = -1;
    pivot.value = limits::Max<Real>();
    const Int mLocal = x.LocalHeight();
    for( Int iLoc=0; iLoc<mLocal; ++iLoc )
    {
        const Real value = x.GetLocal(iLoc,0);
        if( value < pivot.value )
        {
            pivot.value = value;
            pivot.index = x.GlobalRow(iLoc);
        }
    }
    pivot = mpi::AllReduce( pivot, mpi::MinLocOp<Real>(), x.Grid().Comm() );
    return pivot;
}

template<typename Real,
         typename/*=EnableIf<IsReal<Real>>*/>
Entry<Real> MinLoc( const Matrix<Real>& A )
{
    EL_DEBUG_CSE
    const Int m = A.Height();
    const Int n = A.Width();
    const Real* ABuf = A.LockedBuffer();
    const Int ALDim = A.LDim();

    Entry<Real> pivot;
    pivot.i = -1;
    pivot.j = -1;
    pivot.value = limits::Max<Real>();
    for( Int j=0; j<n; ++j )
    {
        for( Int i=0; i<m; ++i )
        {
            const Real value = ABuf[i+j*ALDim];
            if( value < pivot.value )
            {
                pivot.i = i;
                pivot.j = j;
                pivot.value = value;
            }
        }
    }
    return pivot;
}

template<typename Real,
         typename/*=EnableIf<IsReal<Real>>*/>
Entry<Real> MinLoc( const AbstractDistMatrix<Real>& A )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if( !A.Grid().InGrid() )
          LogicError("Viewing processes are not allowed");
    )
    const Real* ABuf = A.LockedBuffer();
    const Int ALDim = A.LDim();

    Entry<Real> pivot;
    pivot.i = -1;
    pivot.j = -1;
    pivot.value = limits::Max<Real>();
    if( A.Participating() )
    {
        // Store the index/value of the local pivot candidate
        const Int mLocal = A.LocalHeight();
        const Int nLocal = A.LocalWidth();
        for( Int jLoc=0; jLoc<nLocal; ++jLoc )
        {
            const Int j = A.GlobalCol(jLoc);
            for( Int iLoc=0; iLoc<mLocal; ++iLoc )
            {
                const Real value = ABuf[iLoc+jLoc*ALDim];
                if( value < pivot.value )
                {
                    const Int i = A.GlobalRow(iLoc);
                    pivot.i = i;
                    pivot.j = j;
                    pivot.value = value;
                }
            }
        }
        // Compute and store the location of the new pivot
        pivot = mpi::AllReduce
                ( pivot, mpi::MinLocPairOp<Real>(), A.DistComm() );
    }
    mpi::Broadcast( pivot, A.Root(), A.CrossComm() );
    return pivot;
}

template<typename Real,
         typename/*=EnableIf<IsReal<Real>>*/>
Entry<Real> SymmetricMinLoc( UpperOrLower uplo, const Matrix<Real>& A )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if( A.Height() != A.Width() )
          LogicError("A must be square");
    )
    const Int n = A.Width();
    const Real* ABuf = A.LockedBuffer();
    const Int ALDim = A.LDim();

    Entry<Real> pivot;
    pivot.i = -1;
    pivot.j = -1;
    pivot.value = limits::Max<Real>();
    if( uplo == LOWER )
    {
        for( Int j=0; j<n; ++j )
        {
            for( Int i=j; i<n; ++i )
            {
                const Real value = ABuf[i+j*ALDim];
                if( value < pivot.value )
                {
                    pivot.i = i;
                    pivot.j = j;
                    pivot.value = value;
                }
            }
        }
    }
    else
    {
        for( Int j=0; j<n; ++j )
        {
            for( Int i=0; i<=j; ++i )
            {
                const Real value = ABuf[i+j*ALDim];
                if( value < pivot.value )
                {
                    pivot.i = i;
                    pivot.j = j;
                    pivot.value = value;
                }
            }
        }
    }
    return pivot;
}

template<typename Real,
         typename/*=EnableIf<IsReal<Real>>*/>
Entry<Real>
SymmetricMinLoc( UpperOrLower uplo, const AbstractDistMatrix<Real>& A )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if( A.Height() != A.Width() )
          LogicError("A must be square");
      if( !A.Grid().InGrid() )
          LogicError("Viewing processes are not allowed");
    )
    Entry<Real> pivot;
    pivot.i = -1;
    pivot.j = -1;
    pivot.value = limits::Max<Real>();
    if( A.Participating() )
    {
        const Int mLocal = A.LocalHeight();
        const Int nLocal = A.LocalWidth();
        if( uplo == LOWER )
        {
            for( Int jLoc=0; jLoc<nLocal; ++jLoc )
            {
                const Int j = A.GlobalCol(jLoc);
                const Int mLocBefore = A.LocalRowOffset(j);
                for( Int iLoc=mLocBefore; iLoc<mLocal; ++iLoc )
                {
                    const Real value = A.GetLocal(iLoc,jLoc);
                    if( value < pivot.value )
                    {
                        const Int i = A.GlobalRow(iLoc);
                        pivot.i = i;
                        pivot.j = j;
                        pivot.value = value;
                    }
                }
            }
        }
        else
        {
            for( Int jLoc=0; jLoc<nLocal; ++jLoc )
            {
                const Int j = A.GlobalCol(jLoc);
                const Int mLocBefore = A.LocalRowOffset(j+1);
                for( Int iLoc=0; iLoc<mLocBefore; ++iLoc )
                {
                    const Real value = A.GetLocal(iLoc,jLoc);
                    if( value < pivot.value )
                    {
                        const Int i = A.GlobalRow(iLoc);
                        pivot.i = i;
                        pivot.j = j;
                        pivot.value = value;
                    }
                }
            }
        }
        // Compute and store the location of the new pivot
        pivot = mpi::AllReduce
                ( pivot, mpi::MinLocPairOp<Real>(), A.DistComm() );
    }
    mpi::Broadcast( pivot, A.Root(), A.CrossComm() );
    return pivot;
}

#define PROTO(Real) \
  template ValueInt<Real> VectorMinLoc( const Matrix<Real>& x ); \
  template ValueInt<Real> VectorMinLoc( const AbstractDistMatrix<Real>& x ); \
  template ValueInt<Real> VectorMinLoc( const DistMultiVec<Real>& x ); \
  template Entry<Real> MinLoc( const Matrix<Real>& x ); \
  template Entry<Real> MinLoc( const AbstractDistMatrix<Real>& x ); \
  template Entry<Real> SymmetricMinLoc \
  ( UpperOrLower uplo, const Matrix<Real>& A ); \
  template Entry<Real> SymmetricMinLoc \
  ( UpperOrLower uplo, const AbstractDistMatrix<Real>& A );

#define EL_NO_COMPLEX_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El
