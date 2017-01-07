/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include <El/blas_like/level1.hpp>
#include <El/matrices.hpp>

namespace El {

template<typename F> 
void MakeLegendre( Matrix<F>& A )
{
    EL_DEBUG_CSE
    if( A.Height() != A.Width() )
        LogicError("Cannot make a non-square matrix Legendre");
    Zero( A );

    const Int n = A.Width();
    for( Int j=0; j<n-1; ++j )
    {
        const F gamma = F(1) / Pow( F(2)*F(j+1), F(2) );
        const F beta = F(1) / (F(2)*Sqrt(F(1)-gamma));
        A(j+1,j) = beta;
        A(j,j+1) = beta;
    }
}

template<typename F>
void MakeLegendre( AbstractDistMatrix<F>& A )
{
    EL_DEBUG_CSE
    if( A.Height() != A.Width() )
        LogicError("Cannot make a non-square matrix Legendre");
    Zero( A );

    const Int localHeight = A.LocalHeight();
    const Int localWidth = A.LocalWidth();
    auto& ALoc = A.Matrix();
    for( Int jLoc=0; jLoc<localWidth; ++jLoc )
    {
        const Int j = A.GlobalCol(jLoc);
        for( Int iLoc=0; iLoc<localHeight; ++iLoc )
        {
            const Int i = A.GlobalRow(iLoc);
            if( j == i+1 || j == i-1 )
            {
                const Int k = Max( i, j );
                const F gamma = F(1) / Pow( F(2)*F(k), F(2) );
                const F beta = F(1) / (F(2)*Sqrt(F(1)-gamma));
                ALoc(iLoc,jLoc) = beta;
            }
        }
    }
}

template<typename F> 
void Legendre( Matrix<F>& A, Int n )
{
    EL_DEBUG_CSE
    A.Resize( n, n );
    MakeLegendre( A );
}

template<typename F> 
void Legendre( AbstractDistMatrix<F>& A, Int n )
{
    EL_DEBUG_CSE
    A.Resize( n, n );
    MakeLegendre( A );
}

#define PROTO(F) \
  template void Legendre( Matrix<F>& A, Int n ); \
  template void Legendre( AbstractDistMatrix<F>& A, Int n );

#define EL_NO_INT_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El
