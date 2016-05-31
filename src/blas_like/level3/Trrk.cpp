/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include <El/blas_like/level1.hpp>
#include <El/blas_like/level3.hpp>

#include "./Trrk/Local.hpp"
#include "./Trrk/NN.hpp"
#include "./Trrk/NT.hpp"
#include "./Trrk/TN.hpp"
#include "./Trrk/TT.hpp"

namespace El {

template<typename T>
void TrrkInternal
( UpperOrLower uplo, 
  Orientation orientA, Orientation orientB,
  T alpha, const Matrix<T>& A, const Matrix<T>& B,
  T beta,        Matrix<T>& C )
{
    DEBUG_ONLY(CSE cse("TrrkInternal"))
    ScaleTrapezoid( beta, uplo, C );
    if( orientA==NORMAL && orientB==NORMAL )
        trrk::TrrkNN( uplo, alpha, A, B, C );
    else if( orientA==NORMAL )
        trrk::TrrkNT( uplo, orientB, alpha, A, B, C );
    else if( orientB==NORMAL )
        trrk::TrrkTN( uplo, orientA, alpha, A, B, C );
    else
        trrk::TrrkTT( uplo, orientA, orientB, alpha, A, B, C );
}

#ifdef EL_HAVE_MKL
template<typename T,typename=EnableIf<IsBlasScalar<T>>>
void TrrkMKL
( UpperOrLower uplo, 
  Orientation orientA, Orientation orientB,
  T alpha, const Matrix<T>& A, const Matrix<T>& B,
  T beta,        Matrix<T>& C )
{
    DEBUG_ONLY(CSE cse("TrrkMKL"))
    const char uploChar = UpperOrLowerToChar( uplo );
    const char orientAChar = OrientationToChar( orientA );
    const char orientBChar = OrientationToChar( orientB );

    mkl::Trrk 
    ( uploChar, orientAChar, orientBChar,
      alpha, A.LockedBuffer(), A.LDim(),
             B.LockedBuffer(), B.LDim(),
      beta,  C.Buffer(),       C.LDim() );
}
#endif

template<typename T,typename=EnableIf<IsBlasScalar<T>>>
void TrrkHelper
( UpperOrLower uplo, 
  Orientation orientA, Orientation orientB,
  T alpha, const Matrix<T>& A, const Matrix<T>& B,
  T beta,        Matrix<T>& C )
{
    DEBUG_ONLY(CSE cse("TrrkHelper"))
#ifdef EL_HAVE_MKL
    TrrkMKL( uplo, orientA, orientB, alpha, A, B, beta, C );
#else
    TrrkInternal( uplo, orientA, orientB, alpha, A, B, beta, C );
#endif
}

template<typename T,typename=DisableIf<IsBlasScalar<T>>,typename=void>
void TrrkHelper
( UpperOrLower uplo, 
  Orientation orientA, Orientation orientB,
  T alpha, const Matrix<T>& A, const Matrix<T>& B,
  T beta,        Matrix<T>& C )
{
    DEBUG_ONLY(CSE cse("TrrkHelper"))
    TrrkInternal( uplo, orientA, orientB, alpha, A, B, beta, C );
}

template<typename T>
void Trrk
( UpperOrLower uplo, 
  Orientation orientA, Orientation orientB,
  T alpha, const Matrix<T>& A, const Matrix<T>& B,
  T beta,        Matrix<T>& C )
{
    DEBUG_ONLY(CSE cse("Trrk"))
    TrrkHelper( uplo, orientA, orientB, alpha, A, B, beta, C );
}

template<typename T>
void Trrk
( UpperOrLower uplo, Orientation orientA, Orientation orientB,
  T alpha, const ElementalMatrix<T>& A, const ElementalMatrix<T>& B,
  T beta,        ElementalMatrix<T>& C )
{
    DEBUG_ONLY(CSE cse("Trrk"))
    ScaleTrapezoid( beta, uplo, C );
    if( orientA==NORMAL && orientB==NORMAL )
        trrk::TrrkNN( uplo, alpha, A, B, C );
    else if( orientA==NORMAL )
        trrk::TrrkNT( uplo, orientB, alpha, A, B, C );
    else if( orientB==NORMAL )
        trrk::TrrkTN( uplo, orientA, alpha, A, B, C );
    else
        trrk::TrrkTT( uplo, orientA, orientB, alpha, A, B, C );
}

#define PROTO(T) \
  template void Trrk \
  ( UpperOrLower uplo, \
    Orientation orientA, Orientation orientB, \
    T alpha, const Matrix<T>& A, \
             const Matrix<T>& B, \
    T beta,        Matrix<T>& C ); \
  template void Trrk \
  ( UpperOrLower uplo, \
    Orientation orientA, Orientation orientB, \
    T alpha, const ElementalMatrix<T>& A, \
             const ElementalMatrix<T>& B, \
    T beta,        ElementalMatrix<T>& C ); \
  template void LocalTrrk \
   ( UpperOrLower uplo, \
     T alpha, const DistMatrix<T,MC,  STAR>& A, \
              const DistMatrix<T,STAR,MR  >& B, \
     T beta,        DistMatrix<T>& C ); \
  template void LocalTrrk \
  ( UpperOrLower uplo, Orientation orientB, \
    T alpha, const DistMatrix<T,MC,STAR>& A, \
             const DistMatrix<T,MR,STAR>& B, \
    T beta,        DistMatrix<T>& C ); \
  template void LocalTrrk \
  ( UpperOrLower uplo, Orientation orientA, \
    T alpha, const DistMatrix<T,STAR,MC>& A, \
             const DistMatrix<T,STAR,MR>& B, \
    T beta,        DistMatrix<T>& C ); \
  template void LocalTrrk \
  ( UpperOrLower uplo, \
    Orientation orientA, Orientation orientB, \
    T alpha, const DistMatrix<T,STAR,MC  >& A, \
             const DistMatrix<T,MR,  STAR>& B, \
    T beta,        DistMatrix<T>& C );

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El
