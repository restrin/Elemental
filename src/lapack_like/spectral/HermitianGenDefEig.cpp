/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

namespace El {

// Compute eigenvalues
// ===================

template<typename Field>
HermitianEigInfo
HermitianGenDefEig
( Pencil pencil,
  UpperOrLower uplo,
  Matrix<Field>& A,
  Matrix<Field>& B,
  Matrix<Base<Field>>& w,
  const HermitianEigCtrl<Field>& ctrl )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if( A.Height() != A.Width() || B.Height() != B.Width() )
          LogicError("Hermitian matrices must be square.");
    )

    Cholesky( uplo, B );
    if( pencil == AXBX )
        TwoSidedTrsm( uplo, NON_UNIT, A, B );
    else
        TwoSidedTrmm( uplo, NON_UNIT, A, B );
    return HermitianEig( uplo, A, w, ctrl );
}

template<typename Field>
HermitianEigInfo
HermitianGenDefEig
( Pencil pencil,
  UpperOrLower uplo,
  AbstractDistMatrix<Field>& APre,
  AbstractDistMatrix<Field>& BPre,
  AbstractDistMatrix<Base<Field>>& w,
  const HermitianEigCtrl<Field>& ctrl )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      AssertSameGrids( APre, BPre, w );
      if( APre.Height() != APre.Width() || BPre.Height() != BPre.Width() )
          LogicError("Hermitian matrices must be square.");
    )

    DistMatrixReadWriteProxy<Field,Field,MC,MR> AProx( APre ), BProx( BPre );
    auto& A = AProx.Get();
    auto& B = BProx.Get();

    Cholesky( uplo, B );
    if( pencil == AXBX )
        TwoSidedTrsm( uplo, NON_UNIT, A, B );
    else
        TwoSidedTrmm( uplo, NON_UNIT, A, B );
    return HermitianEig( uplo, A, w, ctrl );
}

// Compute eigenpairs
// ==================

template<typename Field>
HermitianEigInfo
HermitianGenDefEig
( Pencil pencil,
  UpperOrLower uplo,
  Matrix<Field>& A,
  Matrix<Field>& B,
  Matrix<Base<Field>>& w,
  Matrix<Field>& X,
  const HermitianEigCtrl<Field>& ctrl )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if( A.Height() != A.Width() || B.Height() != B.Width() )
          LogicError("Hermitian matrices must be square.");
    )

    Cholesky( uplo, B );
    if( pencil == AXBX )
        TwoSidedTrsm( uplo, NON_UNIT, A, B );
    else
        TwoSidedTrmm( uplo, NON_UNIT, A, B );
    auto info = HermitianEig( uplo, A, w, X, ctrl );
    if( pencil == AXBX || pencil == ABX )
    {
        const Orientation orientation = ( uplo==LOWER ? ADJOINT : NORMAL );
        Trsm( LEFT, uplo, orientation, NON_UNIT, Field(1), B, X );
    }
    else /* pencil == BAX */
    {
        const Orientation orientation = ( uplo==LOWER ? NORMAL : ADJOINT );
        Trmm( LEFT, uplo, orientation, NON_UNIT, Field(1), B, X );
    }
    return info;
}

template<typename Field>
HermitianEigInfo
HermitianGenDefEig
( Pencil pencil,
  UpperOrLower uplo,
  AbstractDistMatrix<Field>& APre,
  AbstractDistMatrix<Field>& BPre,
  AbstractDistMatrix<Base<Field>>& w,
  AbstractDistMatrix<Field>& XPre,
  const HermitianEigCtrl<Field>& ctrl )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      AssertSameGrids( APre, BPre, w, XPre );
      if( APre.Height() != APre.Width() || BPre.Height() != BPre.Width() )
          LogicError("Hermitian matrices must be square.");
    )

    DistMatrixReadWriteProxy<Field,Field,MC,MR> AProx( APre ), BProx( BPre );
    DistMatrixWriteProxy<Field,Field,MC,MR> XProx( XPre );
    auto& A = AProx.Get();
    auto& B = BProx.Get();
    auto& X = XProx.Get();

    Cholesky( uplo, B );
    if( pencil == AXBX )
        TwoSidedTrsm( uplo, NON_UNIT, A, B );
    else
        TwoSidedTrmm( uplo, NON_UNIT, A, B );
    auto info = HermitianEig( uplo, A, w, X, ctrl );
    if( pencil == AXBX || pencil == ABX )
    {
        const Orientation orientation = ( uplo==LOWER ? ADJOINT : NORMAL );
        Trsm( LEFT, uplo, orientation, NON_UNIT, Field(1), B, X );
    }
    else /* pencil == BAX */
    {
        const Orientation orientation = ( uplo==LOWER ? NORMAL : ADJOINT );
        Trmm( LEFT, uplo, orientation, NON_UNIT, Field(1), B, X );
    }
    return info;
}

#define PROTO(Field) \
  template HermitianEigInfo HermitianGenDefEig \
  ( Pencil pencil, \
    UpperOrLower uplo, \
    Matrix<Field>& A, \
    Matrix<Field>& B, \
    Matrix<Base<Field>>& w, \
    const HermitianEigCtrl<Field>& ctrl ); \
  template HermitianEigInfo HermitianGenDefEig \
  ( Pencil pencil, \
    UpperOrLower uplo, \
    AbstractDistMatrix<Field>& A, \
    AbstractDistMatrix<Field>& B, \
    AbstractDistMatrix<Base<Field>>& w, \
    const HermitianEigCtrl<Field>& ctrl ); \
  template HermitianEigInfo HermitianGenDefEig \
  ( Pencil pencil, \
    UpperOrLower uplo, \
    Matrix<Field>& A, \
    Matrix<Field>& B, \
    Matrix<Base<Field>>& w, \
    Matrix<Field>& X, \
    const HermitianEigCtrl<Field>& ctrl ); \
  template HermitianEigInfo HermitianGenDefEig \
  ( Pencil pencil, \
    UpperOrLower uplo, \
    AbstractDistMatrix<Field>& A, \
    AbstractDistMatrix<Field>& B, \
    AbstractDistMatrix<Base<Field>>& w, \
    AbstractDistMatrix<Field>& X, \
    const HermitianEigCtrl<Field>& ctrl );

#define EL_NO_INT_PROTO
#define EL_ENABLE_QUAD
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El
