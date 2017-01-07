/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   Copyright (c) 2015-2016, Tim Moon
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/

namespace El {

// Forward declaration
template<typename T>
Base<T> MaxNorm( const Matrix<T>& A );

namespace safemstrsm {

/*   Note: See "Robust Triangular Solves for Use in Condition
 *   Estimation" by Edward Anderson for notation and bounds.
 *   Entries in U are assumed to be less (in magnitude) than 
 *   bigNum.
 */
template<typename F>
void LUNBlock
(       Matrix<F>& U,
  const Matrix<F>& shifts,
        Matrix<F>& X,
        Matrix<F>& scales )
{
    EL_DEBUG_CSE
    typedef Base<F> Real;
    EL_DEBUG_ONLY(
      if( U.Height() != U.Width() )
          LogicError("Triangular matrix must be square");
      if( U.Width() != X.Height() )
          LogicError("Matrix dimensions do not match");
      if( shifts.Height() != X.Width() )
          LogicError("Incompatible number of shifts");
    )
    auto diag = GetDiagonal(U);
    const Int n = U.Height();
    const Int numShifts = shifts.Height();

    auto overflowPair = OverflowParameters<Real>();
    const Real smallNum = overflowPair.first;
    const Real bigNum = overflowPair.second;
    
    const Real oneHalf = Real(1)/Real(2);
    const Real oneQuarter = Real(1)/Real(4);

    // Default scale is 1
    scales.Resize( numShifts, 1 );
    Fill( scales, F(1) );
    
    // Compute infinity norms of columns of U (excluding diagonal)
    Matrix<Real> cNorm( n, 1 );
    cNorm(0) = 0;
    for( Int j=1; j<n; ++j )
    {
        // cNorm[j] = MaxNorm( U(IR(0,j),IR(j)) );
        cNorm(j) = 0;
        for( Int i=0; i<j; ++i )
            cNorm(j) = Max( cNorm(j), Abs(U(i,j)) );
    }

    // Iterate through RHS's
    for( Int j=0; j<numShifts; ++j )
    {
        // Initialize triangular system
        SetDiagonal( U, diag );
        ShiftDiagonal( U, -shifts.Get(j,0) );
        auto xj = X( ALL, IR(j) );
        Real scales_j = Real(1);

        // Determine largest entry of RHS
        Real xjMax = MaxNorm( xj );
        if( xjMax >= bigNum )
        {
            const Real s = oneHalf*bigNum/xjMax;
            xj *= s;
            xjMax *= s;
            scales_j *= s;
        }
        if( xjMax <= smallNum )
        {
            continue;
        }

        // Estimate growth of entries in triangular solve
        //   Note: See "Robust Triangular Solves for Use in Condition
        //   Estimation" by Edward Anderson for explanation of bounds.
        Real invGi = 1/xjMax;
        Real invMi = invGi;
        for( Int i=n-1; i>=0; --i )
        {
            const Real absUii = SafeAbs( U(i,i) );
            if( invGi<=smallNum || invMi<=smallNum || absUii<=smallNum )
            {
                invGi = 0;
                break;
            }
            invMi = Min( invMi, absUii*invGi );
            if( i > 0 )
            {
                invGi *= absUii/(absUii+cNorm(i));
            }
        }
        invGi = Min( invGi, invMi );

        if( invGi > smallNum )
        {
            // Call TRSV since estimated growth is not too large
            blas::Trsv( 'U', 'N', 'N', n, &U(0,0), U.LDim(), &xj(0), 1 );
        }
        else
        {
            // Perform backward substitution since estimated growth is large
            for( Int i=n-1; i>=0; --i )
            {
                // Perform division and check for overflow
                const F Uii = U(i,i);
                const Real absUii = SafeAbs( Uii );
                F Xij = xj(i);
                Real absXij = SafeAbs( Xij );
                if( absUii > smallNum )
                {
                    if( absUii<=1 && absXij>=absUii*bigNum )
                    {
                        // Set overflowing entry to 0.5/U[i,i]
                        const Real s = oneHalf/absXij;
                        Xij *= s;
                        xj *= s;
                        xjMax *= s;
                        scales_j *= s;
                    }
                    Xij /= Uii;
                }
                else if( absUii > 0 )
                {
                    if( absXij >= absUii*bigNum )
                    {
                        // Set overflowing entry to bigNum/2
                        const Real s = oneHalf*absUii*bigNum/absXij;
                        Xij *= s;
                        xj *= s;
                        xjMax *= s;
                        scales_j *= s;
                    }
                    Xij /= Uii;
                }
                else
                {
                    // TODO: maybe this tolerance should be loosened to
                    //   | Xij | >= || A || * eps
                    if( absXij >= smallNum )
                    {
                        Xij = F(1);
                        Zero( xj );
                        xjMax = Real(0);
                        scales_j = Real(0);
                    }
                }
                xj(i) = Xij;
                
                if( i > 0 )
                {

                    // Check for possible overflows in AXPY
                    // Note: G(i+1) <= G(i) + | Xij | * cNorm(i)
                    absXij = SafeAbs( Xij );
                    const Real cNorm_i = cNorm(i);
                    if( absXij >= Real(1) &&
                        cNorm_i >= (bigNum-xjMax)/absXij )
                    {
                        const Real s = oneQuarter/absXij;
                        Xij *= s;
                        xj *= s;
                        xjMax *= s;
                        absXij *= s;
                        scales_j *= s;
                    }
                    else if( absXij < Real(1) &&
                             absXij*cNorm_i >= bigNum-xjMax )
                    {
                        const Real s = oneQuarter;
                        Xij *= s;
                        xj *= s;
                        xjMax *= s;
                        absXij *= s;
                        scales_j *= s;
                    }
                    xjMax += absXij*cNorm_i;

                    // AXPY X(0:i,j) -= Xij*U(0:i,i)
                    blas::Axpy( i, -Xij, &U(0,i), 1, &xj(0), 1 );
                }
            }
        }
        scales(j) = scales_j;
    }

    // Reset matrix diagonal
    SetDiagonal( U, diag );
}

/*   Note: See "Robust Triangular Solves for Use in Condition
 *   Estimation" by Edward Anderson for notation and bounds.
 *   Entries in U are assumed to be less (in magnitude) than 
 *   bigNum.
 */
template<typename F>
void LUN
(       Matrix<F>& U,
  const Matrix<F>& shifts,
        Matrix<F>& X,
        Matrix<F>& scales ) 
{
    EL_DEBUG_CSE
    typedef Base<F> Real;

    EL_DEBUG_ONLY(
      if( U.Height() != U.Width() )
          LogicError("Triangular matrix must be square");
      if( U.Width() != X.Height() )
          LogicError("Matrix dimensions do not match");
      if( shifts.Height() != X.Width() )
          LogicError("Incompatible number of shifts");
    )
    const Int m = X.Height();
    const Int n = X.Width();
    const Int bsize = Blocksize();
    const Int kLast = LastOffset( m, bsize );

    const Real oneHalf = Real(1)/Real(2);

    auto overflowPair = OverflowParameters<Real>();
    const Real smallNum = overflowPair.first;
    const Real bigNum = overflowPair.second;

    EL_DEBUG_ONLY(
      if( MaxNorm(U) >= bigNum )
          LogicError("Entries in matrix are too large");
    )
    
    scales.Resize( n, 1 );
    Fill( scales, F(1) );

    Matrix<F> scalesUpdate( n, 1 );

    // Determine largest entry of each RHS
    Matrix<Real> XMax( n, 1 );
    for( Int j=0; j<n; ++j )
    {
        auto xj = X( ALL, IR(j) );
        Real xjMax = MaxNorm( xj );
        if( xjMax >= bigNum )
        {
            const Real s = oneHalf*bigNum/xjMax;
            xj *= s;
            xjMax *= s;
            scales(j) *= s;
        }
        xjMax = Max( xjMax, 2*smallNum );
        XMax(j) = xjMax;
    }
        
    // Perform block triangular solve
    for( Int k=kLast; k>=0; k-=bsize )
    {
        const Int nb = Min(bsize,m-k);

        const Range<Int> ind0( 0,    k    ),
                         ind1( k,    k+nb ),
                         ind2( k+nb, END  );

        auto U01 = U( ind0, ind1 );
        auto U11 = U( ind1, ind1 );

        auto X0 = X( ind0, ALL );
        auto X1 = X( ind1, ALL );
        auto X2 = X( ind2, ALL );

        // Perform triangular solve on diagonal block
        LUNBlock( U11, shifts, X1, scalesUpdate );

        // Apply scalings on RHS
        for( Int j=0; j<n; ++j )
        {
            const Real sj = scalesUpdate.GetRealPart(j,0);
            if( sj < Real(1) )
            {
                scales(j) *= sj;
                auto X0j = X0( ALL, IR(j) );
                auto X2j = X2( ALL, IR(j) );
                X0j *= sj;
                X2j *= sj;
                XMax(j) *= sj;
            }
        }

        if( k > 0 )
        {

            // Compute infinity norms of columns in U01
            // Note: nb*cNorm is the sum of infinity norms
            Real cNorm = 0;
            for( Int j=0; j<nb; ++j )
            {
                cNorm += MaxNorm( U01(ALL,IR(j)) ) / nb;
            }

            // Check for possible overflows in GEMM
            // Note: G(i+1) <= G(i) + nb*cNorm*|| X1[:,j] ||_infty
            for( Int j=0; j<n; ++j )
            {
                auto xj = X( ALL, IR(j) );
                Real xjMax = XMax(j);
                Real X1Max = MaxNorm( X1(ALL,IR(j)) );
                if( X1Max >= 1 &&
                    cNorm >= (bigNum-xjMax)/X1Max/nb )
                {
                    const Real s = oneHalf/(X1Max*nb);
                    scales(j) *= s;
                    xj *= s;
                    xjMax *= s;
                    X1Max *= s;
                }
                else if( X1Max < 1 &&
                         cNorm*X1Max >= (bigNum-xjMax)/nb )
                {
                    const Real s = oneHalf/nb;
                    scales(j) *= s;
                    xj *= s;
                    xjMax *= s;
                    X1Max *= s;
                }
                xjMax += nb*cNorm*X1Max;
                XMax(j) = xjMax;
            }

            // Update RHS with GEMM
            Gemm( NORMAL, NORMAL, F(-1), U01, X1, F(1), X0 );
        }
    }
}

template<typename F>
void LUN
( const AbstractDistMatrix<F>& UPre, 
  const AbstractDistMatrix<F>& shiftsPre,
        AbstractDistMatrix<F>& XPre,
        AbstractDistMatrix<F>& scalesPre ) 
{
    EL_DEBUG_CSE
    typedef Base<F> Real;
  
    EL_DEBUG_ONLY(
      if( UPre.Height() != UPre.Width() )
          LogicError("Triangular matrix must be square");
      if( UPre.Width() != XPre.Height() )
          LogicError("Matrix dimensions do not match");
      if( shiftsPre.Height() != XPre.Width() )
          LogicError("Incompatible number of shifts");
    )

    const Int m = XPre.Height();
    const Int n = XPre.Width();
    const Int bsize = Blocksize();
    const Int kLast = LastOffset( m, bsize );

    DistMatrixReadProxy<F,F,MC,MR> UProx( UPre );
    DistMatrixReadProxy<F,F,VR,STAR> shiftsProx( shiftsPre );
    DistMatrixReadWriteProxy<F,F,MC,MR> XProx( XPre );
    DistMatrixWriteProxy<F,F,VR,STAR> scalesProx( scalesPre );
    auto& U = UProx.GetLocked();
    auto& shifts = shiftsProx.GetLocked();
    auto& X = XProx.Get();
    auto& scales = scalesProx.Get();
    
    const Grid& g = U.Grid();
    DistMatrix<F,MC,  STAR> U01_MC_STAR(g);
    DistMatrix<F,STAR,STAR> U11_STAR_STAR(g);
    DistMatrix<F,STAR,MR  > X1_STAR_MR(g);
    DistMatrix<F,STAR,VR  > X1_STAR_VR(g);
    DistMatrix<F,VR,  STAR> scalesUpdate_VR_STAR(g);
    DistMatrix<F,MR,  STAR> scalesUpdate_MR_STAR( X.Grid() );

    scales.Resize( n, 1 );
    Fill( scales, F(1) );

    scalesUpdate_VR_STAR.Resize( n, 1 );

    const Int XLocalWidth = X.LocalWidth();
    
    for( Int k=kLast; k>=0; k-=bsize )
    {
        const Int nb = Min(bsize,m-k);

        const Range<Int> ind0( 0   , k    ),
                         ind1( k   , k+nb ),
                         ind2( k+nb, END  );

        auto U01 = U( ind0, ind1 );
        auto U11 = U( ind1, ind1 );

        auto X0 = X( ind0, ALL );
        auto X1 = X( ind1, ALL );
        auto X2 = X( ind2, ALL );

        // Perform triangular solve on diagonal block
        // X1[* ,VR] := U11^-1[* ,* ] X1[* ,VR]
        U11_STAR_STAR = U11; // U11[* ,* ] <- U11[MC,MR]
        X1_STAR_VR.AlignWith( shifts );
        X1_STAR_VR = X1; // X1[* ,VR] <- X1[MC,MR]
        scalesUpdate_VR_STAR.AlignWith( shifts );
        LUN
        ( U11_STAR_STAR.Matrix(), shifts.LockedMatrix(), 
          X1_STAR_VR.Matrix(), scalesUpdate_VR_STAR.Matrix() );
        X1_STAR_MR.AlignWith( X0 );
        X1_STAR_MR = X1_STAR_VR; // X1[* ,MR]  <- X1[* ,VR]
        X1 = X1_STAR_MR; // X1[MC,MR] <- X1[* ,MR]

        // Apply scalings on RHS
        scalesUpdate_MR_STAR.AlignWith( X );
        scalesUpdate_MR_STAR = scalesUpdate_VR_STAR;
        for( Int jLoc=0; jLoc<XLocalWidth; ++jLoc )
        {
            const Real sigma = scalesUpdate_MR_STAR.GetLocalRealPart(jLoc,0);
            if( sigma < Real(1) )
            {
                // X1 has already been rescaled, but X0 and X2 have not
                blas::Scal( X0.LocalHeight(), sigma, X0.Buffer(0,jLoc), 1 );
                // TODO: Delay rescaling of X2 until the end since it is no
                //       longer used within this loop?
                blas::Scal( X2.LocalHeight(), sigma, X2.Buffer(0,jLoc), 1 );
            }
        }
        DiagonalScale( LEFT,  NORMAL, scalesUpdate_VR_STAR, scales );
        
        if( k > 0 )
        {
            // Update RHS with GEMM
            // X0[MC,MR] -= U01[MC,* ] X1[* ,MR]
            U01_MC_STAR.AlignWith( X0 );
            U01_MC_STAR = U01; // U01[MC,* ] <- U01[MC,MR]
            LocalGemm
            ( NORMAL, NORMAL, F(-1), U01_MC_STAR, X1_STAR_MR, F(1), X0 );
        }
    }
}
  
} // namespace safemstrsm
} // namespace El
