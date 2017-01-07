/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_QR_PANEL_HPP
#define EL_QR_PANEL_HPP

namespace El {
namespace qr {

template<typename F>
void PanelHouseholder
( Matrix<F>& A,
  Matrix<F>& householderScalars,
  Matrix<Base<F>>& signature )
{
    EL_DEBUG_CSE
    typedef Base<F> Real;
    const Int m = A.Height();
    const Int n = A.Width();
    const Int minDim = Min(m,n);
    householderScalars.Resize( minDim, 1 );
    signature.Resize( minDim, 1 );

    Matrix<F> z21;

    for( Int k=0; k<minDim; ++k )
    {
        const Range<Int> ind1( k ), ind2( k+1, END ), indB( k, END );

        auto alpha11 = A( ind1, ind1 );
        auto a21     = A( ind2, ind1 );
        auto aB1     = A( indB, ind1 );
        auto AB2     = A( indB, ind2 );

        // Find tau and u such that
        //  / I - tau | 1 | | 1, u^H | \ | alpha11 | = | beta |
        //  \         | u |            / |     a21 | = |    0 |
        const F tau = LeftReflector( alpha11, a21 );
        householderScalars(k) = tau;

        // Temporarily set aB1 = | 1 |
        //                       | u |
        const F alpha = alpha11(0);
        alpha11(0) = 1;

        // AB2 := Hous(aB1,tau) AB2
        //      = (I - tau aB1 aB1^H) AB2
        //      = AB2 - tau aB1 (AB2^H aB1)^H
        Zeros( z21, AB2.Width(), 1 );
        Gemv( ADJOINT, F(1), AB2, aB1, F(0), z21 );
        Ger( -tau, aB1, z21, AB2 );

        // Replace alpha11's value
        alpha11(0) = alpha;
    }
    // Form d and rescale R
    auto R = A( IR(0,minDim), ALL );
    GetRealPartOfDiagonal(R,signature);
    auto sgn = []( const Real& delta )
               { return delta >= Real(0) ? Real(1) : Real(-1); };
    EntrywiseMap( signature, MakeFunction(sgn) );
    DiagonalScaleTrapezoid( LEFT, UPPER, NORMAL, signature, R );
}

template<typename F>
void PanelHouseholder
( DistMatrix<F>& A,
  AbstractDistMatrix<F>& householderScalars,
  AbstractDistMatrix<Base<F>>& signature )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(AssertSameGrids( A, householderScalars, signature ))
    typedef Base<F> Real;
    const Grid& g = A.Grid();
    DistMatrix<F,MC,STAR> aB1_MC_STAR(g);
    DistMatrix<F,MR,STAR> z21_MR_STAR(g);

    const Int m = A.Height();
    const Int n = A.Width();
    const Int minDim = Min(m,n);

    if( householderScalars.Height() != minDim ||
        householderScalars.Width() != 1 )
        LogicError("Unexpected size of householderScalars");
    if( signature.Height() != minDim || signature.Width() != 1 )
        LogicError("Unexpected size of signature");

    householderScalars.Resize( minDim, 1 );

    for( Int k=0; k<minDim; ++k )
    {
        const Range<Int> ind1( k ), ind2( k+1, END ), indB( k, END );

        auto alpha11 = A( ind1, ind1 );
        auto a21     = A( ind2, ind1 );
        auto aB1     = A( indB, ind1 );
        auto AB2     = A( indB, ind2 );

        // Find tau and u such that
        //  / I - tau | 1 | | 1, u^H | \ | alpha11 | = | beta |
        //  \         | u |            / |     a21 | = |    0 |
        const F tau = LeftReflector( alpha11, a21 );
        householderScalars.Set( k, 0, tau );

        // Temporarily set aB1 = | 1 |
        //                       | u |
        F alpha = 0;
        if( alpha11.IsLocal(0,0) )
        {
            alpha = alpha11.GetLocal(0,0);
            alpha11.SetLocal(0,0,F(1));
        }

        // AB2 := Hous(aB1,tau) AB2
        //      = (I - tau aB1 aB1^H) AB2
        //      = AB2 - tau aB1 (AB2^H aB1)^H
        aB1_MC_STAR.AlignWith( AB2 );
        aB1_MC_STAR = aB1;
        z21_MR_STAR.AlignWith( AB2 );
        Zeros( z21_MR_STAR, AB2.Width(), 1 );
        LocalGemv( ADJOINT, F(1), AB2, aB1_MC_STAR, F(0), z21_MR_STAR );
        El::AllReduce( z21_MR_STAR, AB2.ColComm() );
        Ger
        ( -tau, aB1_MC_STAR.LockedMatrix(), z21_MR_STAR.LockedMatrix(),
          AB2.Matrix() );

        // Replace alpha11's value
        if( alpha11.IsLocal(0,0) )
            alpha11.SetLocal(0,0,alpha);
    }
    // Form d and rescale R
    auto R = A( IR(0,minDim), ALL );
    GetRealPartOfDiagonal(R,signature);
    auto sgn = []( const Real& delta )
               { return delta >= Real(0) ? Real(1) : Real(-1); };
    EntrywiseMap( signature, MakeFunction(sgn) );
    DiagonalScaleTrapezoid( LEFT, UPPER, NORMAL, signature, R );
}

} // namespace qr
} // namespace El

#endif // ifndef EL_QR_PANEL_HPP
