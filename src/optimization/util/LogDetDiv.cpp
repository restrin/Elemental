/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

namespace El {

template<typename Field>
Base<Field> LogDetDiv
( UpperOrLower uplo,
  const Matrix<Field>& A,
  const Matrix<Field>& B )
{
    EL_DEBUG_CSE
    if( A.Height() != A.Width() || B.Height() != B.Width() ||
        A.Height() != B.Height() )
        LogicError("A and B must be square matrices of the same size");

    typedef Base<Field> Real;
    const Int n = A.Height();

    Matrix<Field> ACopy( A ), BCopy( B );
    Cholesky( uplo, ACopy );
    Cholesky( uplo, BCopy );

    if( uplo == LOWER )
    {
        Trstrm( LEFT, uplo, NORMAL, NON_UNIT, Field(1), BCopy, ACopy );
    }
    else
    {
        MakeTrapezoidal( uplo, ACopy );
        Trsm( LEFT, uplo, NORMAL, NON_UNIT, Field(1), BCopy, ACopy );
    }

    MakeTrapezoidal( uplo, ACopy );
    const Real frobNorm = FrobeniusNorm( ACopy );

    Matrix<Field> d;
    GetDiagonal( ACopy, d );
    Real logDet(0);
    for( Int i=0; i<n; ++i )
        logDet += 2*Log( RealPart(d.Get(i,0)) );

    return frobNorm*frobNorm - logDet - Real(n);
}

template<typename Field>
Base<Field> LogDetDiv
( UpperOrLower uplo,
  const AbstractDistMatrix<Field>& A,
  const AbstractDistMatrix<Field>& B )
{
    EL_DEBUG_CSE
    AssertSameGrids( A, B );
    if( A.Height() != A.Width() || B.Height() != B.Width() ||
        A.Height() != B.Height() )
        LogicError("A and B must be square matrices of the same size");

    typedef Base<Field> Real;
    const Int n = A.Height();
    const Grid& g = A.Grid();

    DistMatrix<Field> ACopy( A ), BCopy( B );
    Cholesky( uplo, ACopy );
    Cholesky( uplo, BCopy );

    if( uplo == LOWER )
    {
        Trstrm( LEFT, uplo, NORMAL, NON_UNIT, Field(1), BCopy, ACopy );
    }
    else
    {
        MakeTrapezoidal( uplo, ACopy );
        Trsm( LEFT, uplo, NORMAL, NON_UNIT, Field(1), BCopy, ACopy );
    }

    MakeTrapezoidal( uplo, ACopy );
    const Real frobNorm = FrobeniusNorm( ACopy );

    Real localLogDet(0);
    DistMatrix<Field,MD,STAR> d(g);
    GetDiagonal( ACopy, d );
    if( d.Participating() )
    {
        const Int nLocalDiag = d.LocalHeight();
        for( Int iLocal=0; iLocal<nLocalDiag; ++iLocal )
        {
            const Real delta = RealPart(d.GetLocal(iLocal,0));
            localLogDet += 2*Log(delta);
        }
    }
    const Real logDet = mpi::AllReduce( localLogDet, g.VCComm() );
    return frobNorm*frobNorm - logDet - Real(n);
}

#define PROTO(Field) \
  template Base<Field> LogDetDiv \
  ( UpperOrLower uplo, const Matrix<Field>& A, const Matrix<Field>& B ); \
  template Base<Field> LogDetDiv \
  ( UpperOrLower uplo, \
    const AbstractDistMatrix<Field>& A, const AbstractDistMatrix<Field>& B );

#define EL_NO_INT_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El
