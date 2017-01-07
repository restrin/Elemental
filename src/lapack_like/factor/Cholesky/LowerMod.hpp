/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_CHOLESKY_LOWER_MOD_HPP
#define EL_CHOLESKY_LOWER_MOD_HPP

// TODO: Blocked algorithms
namespace El {
namespace cholesky {

namespace mod {

template<typename F>
void LowerUpdate( Matrix<F>& L, Matrix<F>& V )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if( L.Height() != L.Width() )
          LogicError("Cholesky factors must be square");
      if( V.Height() != L.Height() )
          LogicError("V is the wrong height");
    )
    const Int m = V.Height();

    Matrix<F> z21;

    for( Int k=0; k<m; ++k )
    {
        const IR ind1( k ), ind2( k+1, END );

        F& lambda11 = L(k,k);
        auto l21 = L( ind2, ind1 );

        auto v1 = V( ind1, ALL );
        auto V2 = V( ind2, ALL );

        // Find tau and u such that
        //  | lambda11 u | /I - tau | 1   | | 1 conj(u) |\ = | -beta 0 |
        //                 \        | u^T |              /
        // where beta >= 0
        const F tau = RightReflector( lambda11, v1 );

        // Apply the negative Householder reflector from the right:
        // | l21 V2 | := -| l21 V2 | + tau | l21 V2 | | 1   | | 1 conj(u) |
        //                                           | u^T |
        //             = -| l21 V2 | + tau (l21 + V2 u^T) | 1 conj(u) |
        lambda11 = -lambda11;
        z21 = l21;
        Gemv( NORMAL, F(1), V2, v1, F(1), z21 );
        l21 *= -1;
        Axpy( tau, z21, l21 );
        V2 *= -1;
        Ger( tau, z21, v1, V2 );
    }
}

template<typename F>
void LowerUpdate
( AbstractDistMatrix<F>& LPre,
  AbstractDistMatrix<F>& VPre )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if( LPre.Height() != LPre.Width() )
          LogicError("Cholesky factors must be square");
      if( VPre.Height() != LPre.Height() )
          LogicError("V is the wrong height");
      AssertSameGrids( LPre, VPre );
    )

    DistMatrixReadWriteProxy<F,F,MC,MR> LProx( LPre ), VProx( VPre );
    auto& L = LProx.Get();
    auto& V = VProx.Get();

    const Int m = V.Height();
    const Grid& grid = L.Grid();
    DistMatrix<F,MC,STAR> z21_MC_STAR(grid), b21_MC_STAR(grid);
    DistMatrix<F,STAR,MR> v1_STAR_MR(grid);

    for( Int k=0; k<m; ++k )
    {
        const IR ind1( k ), ind2( k+1, END );

        F lambda11 = L.Get( k, k );
        auto l21 = L( ind2, ind1 );

        auto v1 = V( ind1, ALL );
        auto V2 = V( ind2, ALL );

        // Find tau and u such that
        //  | lambda11 u | /I - tau | 1   | | 1 conj(u) |\ = | -beta 0 |
        //                 \        | u^T |              /
        // where beta >= 0
        const F tau = RightReflector( lambda11, v1 );

        // Apply the negative Householder reflector from the right:
        // | l21 V2 | := -| l21 V2 | + tau | l21 V2 | | 1   | | 1 conj(u) |
        //                                            | u^T |
        //             = -| l21 V2 | + tau (l21 + V2 u^T) | 1 conj(u) |
        L.Set( k, k, -lambda11 );
        z21_MC_STAR.AlignWith( V2 );
        b21_MC_STAR.AlignWith( V2 );
        v1_STAR_MR.AlignWith( V2 );
        z21_MC_STAR = l21;
        v1_STAR_MR = v1;
        Zeros( b21_MC_STAR, V2.Height(), 1 );
        LocalGemv( NORMAL, F(1), V2, v1_STAR_MR, F(0), b21_MC_STAR );
        El::AllReduce( b21_MC_STAR, V2.RowComm() );
        z21_MC_STAR += b21_MC_STAR;
        l21 *= -1;
        Axpy( tau, z21_MC_STAR, l21 );
        V2 *= -1;
        LocalGer( tau, z21_MC_STAR, v1_STAR_MR, V2 );
    }
}

template<typename F>
void LowerDowndate( Matrix<F>& L, Matrix<F>& V )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if( L.Height() != L.Width() )
          LogicError("Cholesky factors must be square");
      if( V.Height() != L.Height() )
          LogicError("V is the wrong height");
    )
    const Int m = V.Height();

    Matrix<F> z21;

    for( Int k=0; k<m; ++k )
    {
        const IR ind1( k ), ind2( k+1, END );

        F& lambda11 = L(k,k);
        auto l21 = L( ind2, ind1 );

        auto v1 = V( ind1, ALL );
        auto V2 = V( ind2, ALL );

        // Find tau and u such that
        //  | lambda11 u | /I - 1/tau Sigma | 1   | | 1 conj(u) |\ = | -beta 0 |
        //                 \                | u^T |              /
        // where Sigma = diag(+1,-1,...,-1) and beta >= 0
        const F tau = RightHyperbolicReflector( lambda11, v1 );

        // Apply the negative of the hyperbolic Householder reflector from the
        // right:
        // |l21 V2| := -|l21 V2| + 1/tau |l21 V2| Sigma |1  | |1 conj(u)|
        //                                              |u^T|
        //           = -|l21 V2| + 1/tau |l21 -V2| |1  | |1 conj(u)|
        //                                         |u^T|
        //           = -|l21 V2| + 1/tau |l21 - V2 u^T| |1 conj(u)|
        lambda11 = -lambda11;
        z21 = l21;
        Gemv( NORMAL, F(-1), V2, v1, F(1), z21 );
        l21 *= -1;
        V2 *= -1;
        Axpy( F(1)/tau, z21, l21 );
        Ger( F(1)/tau, z21, v1, V2 );
    }
}

template<typename F>
void LowerDowndate( AbstractDistMatrix<F>& LPre, AbstractDistMatrix<F>& VPre )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if( LPre.Height() != LPre.Width() )
          LogicError("Cholesky factors must be square");
      if( VPre.Height() != LPre.Height() )
          LogicError("V is the wrong height");
      AssertSameGrids( LPre, VPre );
    )

    DistMatrixReadWriteProxy<F,F,MC,MR> LProx( LPre ), VProx( VPre );
    auto& L = LProx.Get();
    auto& V = VProx.Get();

    const Int m = V.Height();
    const Grid& grid = L.Grid();
    DistMatrix<F,MC,STAR> z21_MC_STAR(grid), b21_MC_STAR(grid);
    DistMatrix<F,STAR,MR> v1_STAR_MR(grid);

    for( Int k=0; k<m; ++k )
    {
        const IR ind1( k ), ind2( k+1, END );

        F lambda11 = L.Get( k, k );
        auto l21 = L( ind2, ind1 );

        auto v1 = V( ind1, ALL );
        auto V2 = V( ind2, ALL );

        // Find tau and u such that
        //  | lambda11 u | /I - 1/tau Sigma | 1   | | 1 conj(u) |\ = | -beta 0 |
        //                 \                | u^T |              /
        // where Sigma = diag(+1,-1,...,-1) and beta >= 0
        const F tau = RightHyperbolicReflector( lambda11, v1 );

        // Apply the negative of the hyperbolic Householder reflector from the
        // right:
        // |l21 V2| := -|l21 V2| + 1/tau |l21 V2| Sigma |1  | |1 conj(u)|
        //                                              |u^T|
        //           = -|l21 V2| + 1/tau |l21 -V2| |1  | |1 conj(u)|
        //                                         |u^T|
        //           = -|l21 V2| + 1/tau |l21 - V2 u^T| |1 conj(u)|
        L.Set( k, k, -lambda11 );
        z21_MC_STAR.AlignWith( V2 );
        b21_MC_STAR.AlignWith( V2 );
        v1_STAR_MR.AlignWith( V2 );
        z21_MC_STAR = l21;
        v1_STAR_MR = v1;
        Zeros( b21_MC_STAR, V2.Height(), 1 );
        LocalGemv( NORMAL, F(-1), V2, v1_STAR_MR, F(0), b21_MC_STAR );
        El::AllReduce( b21_MC_STAR, V2.RowComm() );
        z21_MC_STAR += b21_MC_STAR;
        l21 *= -1;
        V2 *= -1;
        Axpy( F(1)/tau, z21_MC_STAR, l21 );
        LocalGer( F(1)/tau, z21_MC_STAR, v1_STAR_MR, V2 );
    }
}

} // namespace mod

template<typename F>
void LowerMod( Matrix<F>& L, Base<F> alpha, Matrix<F>& V )
{
    EL_DEBUG_CSE
    typedef Base<F> Real;
    if( alpha == Real(0) )
        return;
    else if( alpha > Real(0) )
    {
        V *= Sqrt(alpha);
        mod::LowerUpdate( L, V );
    }
    else
    {
        V *= Sqrt(-alpha);
        mod::LowerDowndate( L, V );
    }
}

template<typename F>
void LowerMod
( AbstractDistMatrix<F>& L,
  Base<F> alpha,
  AbstractDistMatrix<F>& V )
{
    EL_DEBUG_CSE
    typedef Base<F> Real;
    if( alpha == Real(0) )
        return;
    else if( alpha > Real(0) )
    {
        V *= Sqrt(alpha);
        mod::LowerUpdate( L, V );
    }
    else
    {
        V *= Sqrt(-alpha);
        mod::LowerDowndate( L, V );
    }
}

} // namespace cholesky
} // namespace El

#endif // ifndef EL_CHOLESKY_LOWER_MOD_HPP
