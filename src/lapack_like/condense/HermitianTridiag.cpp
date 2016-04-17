/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

#include "./HermitianTridiag/L.hpp"
#include "./HermitianTridiag/LSquare.hpp"
#include "./HermitianTridiag/U.hpp"
#include "./HermitianTridiag/USquare.hpp"

#include "./HermitianTridiag/ApplyQ.hpp"

namespace El {

template<typename F>
void HermitianTridiag( UpperOrLower uplo, Matrix<F>& A, Matrix<F>& t )
{
    DEBUG_ONLY(CSE cse("HermitianTridiag"))
    if( uplo == LOWER )
        herm_tridiag::L( A, t );
    else
        herm_tridiag::U( A, t );
}

template<typename F> 
void HermitianTridiag
( UpperOrLower uplo,
  ElementalMatrix<F>& APre,
  ElementalMatrix<F>& tPre,
  const HermitianTridiagCtrl<F>& ctrl )
{
    DEBUG_ONLY(CSE cse("HermitianTridiag"))

    DistMatrixReadWriteProxy<F,F,MC,MR> AProx( APre );
    DistMatrixWriteProxy<F,F,STAR,STAR> tProx( tPre );
    auto& A = AProx.Get();
    auto& t = tProx.Get();

    const Grid& g = A.Grid();
    if( ctrl.approach == HERMITIAN_TRIDIAG_NORMAL )
    {
        // Use the pipelined algorithm for nonsquare meshes
        if( uplo == LOWER )
            herm_tridiag::L( A, t, ctrl.symvCtrl );
        else
            herm_tridiag::U( A, t, ctrl.symvCtrl );
    }
    else if( ctrl.approach == HERMITIAN_TRIDIAG_SQUARE )
    {
        // Drop down to a square mesh 
        const Int p = g.Size();
        const Int pSqrt = Int(sqrt(double(p)));

        vector<int> squareRanks(pSqrt*pSqrt);
        if( ctrl.order == g.Order() )
        {
            for( Int j=0; j<pSqrt; ++j )
                for( Int i=0; i<pSqrt; ++i )
                    squareRanks[i+j*pSqrt] = i+j*pSqrt;
        }
        else
        {
            for( Int j=0; j<pSqrt; ++j )
                for( Int i=0; i<pSqrt; ++i )
                    squareRanks[i+j*pSqrt] = j+i*pSqrt;
        }

        mpi::Group owningGroup = g.OwningGroup();
        mpi::Group squareGroup;
        mpi::Incl
        ( owningGroup, squareRanks.size(), squareRanks.data(), squareGroup );

        mpi::Comm viewingComm = g.ViewingComm();
        const Grid squareGrid( viewingComm, squareGroup, pSqrt );
        DistMatrix<F> ASquare(squareGrid);
        DistMatrix<F,STAR,STAR> tSquare(squareGrid);

        // Perform the fast tridiagonalization on the square grid
        ASquare = A;
        if( ASquare.Participating() )
        {
            if( uplo == LOWER )
                herm_tridiag::LSquare( ASquare, tSquare, ctrl.symvCtrl );
            else
                herm_tridiag::USquare( ASquare, tSquare, ctrl.symvCtrl ); 
        }
        tSquare.MakeConsistent( true );
        A = ASquare;
        t = tSquare;

        mpi::Free( squareGroup );
    }
    else
    {
        // Use the normal approach unless we're already on a square 
        // grid, in which case we use the fast square method.
        if( g.Height() == g.Width() )
        {
            if( uplo == LOWER )
                herm_tridiag::LSquare( A, t, ctrl.symvCtrl );
            else
                herm_tridiag::USquare( A, t, ctrl.symvCtrl ); 
        }
        else
        {
            if( uplo == LOWER )
                herm_tridiag::L( A, t, ctrl.symvCtrl );
            else
                herm_tridiag::U( A, t, ctrl.symvCtrl );
        }
    }
}

namespace herm_tridiag {

template<typename F>
void ExplicitCondensed( UpperOrLower uplo, Matrix<F>& A )
{
    DEBUG_ONLY(CSE cse("herm_tridiag::ExplicitCondensed"))
    Matrix<F> t;
    HermitianTridiag( uplo, A, t );
    if( uplo == UPPER )
        MakeTrapezoidal( LOWER, A, 1 );
    else
        MakeTrapezoidal( UPPER, A, -1 );
}

template<typename F>
void ExplicitCondensed
( UpperOrLower uplo,
  ElementalMatrix<F>& A, 
  const HermitianTridiagCtrl<F>& ctrl )
{
    DEBUG_ONLY(CSE cse("herm_tridiag::ExplicitCondensed"))
    DistMatrix<F,STAR,STAR> t(A.Grid());
    HermitianTridiag( uplo, A, t, ctrl );
    if( uplo == UPPER )
        MakeTrapezoidal( LOWER, A, 1 );
    else
        MakeTrapezoidal( UPPER, A, -1 );
}

} // namespace herm_tridiag

#define PROTO(F) \
  template void HermitianTridiag \
  ( UpperOrLower uplo, \
    Matrix<F>& A, \
    Matrix<F>& t ); \
  template void HermitianTridiag \
  ( UpperOrLower uplo, \
    ElementalMatrix<F>& A, \
    ElementalMatrix<F>& t, \
    const HermitianTridiagCtrl<F>& ctrl ); \
  template void herm_tridiag::ExplicitCondensed \
  ( UpperOrLower uplo, Matrix<F>& A ); \
  template void herm_tridiag::ExplicitCondensed \
  ( UpperOrLower uplo, \
    ElementalMatrix<F>& A, \
    const HermitianTridiagCtrl<F>& ctrl ); \
  template void herm_tridiag::ApplyQ \
  ( LeftOrRight side, \
    UpperOrLower uplo, \
    Orientation orientation, \
    const Matrix<F>& A, \
    const Matrix<F>& t, \
          Matrix<F>& B ); \
  template void herm_tridiag::ApplyQ \
  ( LeftOrRight side, \
    UpperOrLower uplo, \
    Orientation orientation, \
    const ElementalMatrix<F>& A, \
    const ElementalMatrix<F>& t, \
          ElementalMatrix<F>& B );

#define EL_NO_INT_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El
