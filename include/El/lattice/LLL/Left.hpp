/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_LATTICE_LLL_LEFT_HPP
#define EL_LATTICE_LLL_LEFT_HPP

namespace El {
namespace lll {

// Put the k'th column of B into the k'th column of QR and then rotate
// said column with the first k-1 (scaled) Householder reflectors.

template<typename Z, typename F>
void ExpandQR
( Int k,
  const Matrix<Z>& B,
        Matrix<F>& QR,
  const Matrix<F>& t,
  const Matrix<Base<F>>& d,
  Int numOrthog,
  bool time )
{
    DEBUG_ONLY(CSE cse("lll::ExpandQR"))
    typedef Base<F> Real;
    const Int m = B.Height();
    const Int n = B.Width();
    const Int minDim = Min(m,n);
    const Z* BBuf = B.LockedBuffer();
          F* QRBuf = QR.Buffer();  
    const F* tBuf = t.LockedBuffer();
    const Base<F>* dBuf = d.LockedBuffer();
    const Int BLDim = B.LDim();
    const Int QRLDim = QR.LDim();

    // Copy in the k'th column of B
    for( Int i=0; i<m; ++i )
        QRBuf[i+k*QRLDim] = F(BBuf[i+k*BLDim]);

    if (k == 0)
        return;

    if( time )
        applyHouseTimer.Start();
    for( Int orthog=0; orthog<numOrthog; ++orthog )
    {
        if ( k >= 0)
        {
            Output("ApplyQ for k=", k);
            auto H   = QR( ALL, IR(0,k) );
            auto col = QR( ALL, k );
            auto tt  = t( IR(0,k), ALL );
            auto dd  = d( IR(0,k), ALL );
            qr::ApplyQ(LEFT, ADJOINT, H, tt, dd, col);
        }
        else
        {
            for( Int i=0; i<Min(k,minDim); ++i )
            {
                // Apply the i'th Householder reflector
    
                // Temporarily replace QR(i,i) with 1
                const Real alpha = RealPart(QRBuf[i+i*QRLDim]);
                QRBuf[i+i*QRLDim] = 1;

                const F innerProd =
                  blas::Dot
                  ( m-i,
                    &QRBuf[i+i*QRLDim], 1,
                    &QRBuf[i+k*QRLDim], 1 );
                blas::Axpy
                ( m-i, -tBuf[i]*innerProd,
                  &QRBuf[i+i*QRLDim], 1,
                  &QRBuf[i+k*QRLDim], 1 );

                // Fix the scaling
                QRBuf[i+k*QRLDim] *= dBuf[i];

                // Restore H(i,i)
                QRBuf[i+i*QRLDim] = alpha; 
            }
        }
    }
    if( time )
        applyHouseTimer.Stop();
}

template<typename F>
void HouseholderStep
( Int k,
  Matrix<F>& QR,
  Matrix<F>& t,
  Matrix<Base<F>>& d,
  bool time )
{
    DEBUG_ONLY(CSE cse("lll::HouseholderStep"))
    typedef Base<F> Real;
    const Int m = QR.Height();
    const Int n = QR.Width();
    if( k >= Min(m,n) )
        return;

    if( time )
        houseStepTimer.Start();

    // Perform the next step of Householder reduction
    F* QRBuf = QR.Buffer();
    const Int QRLDim = QR.LDim();
    F& rhokk = QRBuf[k+k*QRLDim]; 
    if( time )
        houseViewTimer.Start();
    auto qr21 = QR( IR(k+1,END), IR(k) );
    if( time )
        houseViewTimer.Stop();
    if( time )
        houseReflectTimer.Start();
    F tau = LeftReflector( rhokk, qr21 );
    if( time )
        houseReflectTimer.Stop();
    t.Set( k, 0, tau );
    if( RealPart(rhokk) < Real(0) )
    {
        d.Set( k, 0, -1 );
        rhokk *= -1;
    }
    else
        d.Set( k, 0, +1 );

    if( time )
        houseStepTimer.Stop();
}

template<typename Z, typename F>
Base<F> Norm2
( Matrix<Z>& B,
  Matrix<F>& bcol,
  Int k )
{
    auto col = B(ALL, IR(k));
    Copy(col, bcol);
    return El::FrobeniusNorm(bcol);
}

// Return true if the new column is a zero vector
template<typename Z, typename F>
bool Step
( Int k,
  Matrix<Z>& B,
  Matrix<Z>& U,
  Matrix<F>& QR,
  Matrix<F>& t,
  Matrix<Base<F>>& d,
  Matrix<F>& bcol,
  Matrix<Base<F>>& colNorms,
  bool formU,
  const LLLCtrl<Base<F>>& ctrl )
{
    DEBUG_ONLY(CSE cse("lll::Step"))
    typedef Base<F> Real;
    const Int m = B.Height();
    const Int n = B.Width();
    const Real eps = limits::Epsilon<Real>();
    const Real thresh = Pow(limits::Epsilon<Real>(), Real(0.5));
    if( ctrl.time )
        stepTimer.Start();

    Z* BBuf = B.Buffer();
    Z* UBuf = U.Buffer();
    F* QRBuf = QR.Buffer();
    const Int BLDim = B.LDim();
    const Int ULDim = U.LDim();
    const Int QRLDim = QR.LDim();
    
    bool colUpdated = false;

    while( true ) 
    {
        if( !ctrl.unsafeSizeReduct && !limits::IsFinite(colNorms.Get(k,0)) )
            RuntimeError("Encountered an unbounded norm; increase precision");
        if( !ctrl.unsafeSizeReduct && colNorms.Get(k,0) > Real(1)/eps )
            RuntimeError("Encountered norm greater than 1/eps, where eps=",eps);

        if( colNorms.Get(k,0) <= ctrl.zeroTol )
        {
            for( Int i=0; i<m; ++i )
                BBuf[i+k*BLDim] = 0;
            for( Int i=0; i<m; ++i )
                QRBuf[i+k*QRLDim] = 0;
            if( k < Min(m,n) )
            {
                t.Set( k, 0, Real(2) );
                d.Set( k, 0, Real(1) );
            }
            if( ctrl.time )
                stepTimer.Stop();
            return true;
        }

        lll::ExpandQR( k, B, QR, t, d, ctrl.numOrthog, ctrl.time );

        if( ctrl.time )
            roundTimer.Start();

        if( ctrl.variant == LLL_WEAK )
        {
            const Real rho_km1_km1 = RealPart(QRBuf[(k-1)+(k-1)*QRLDim]);
            if( rho_km1_km1 > ctrl.zeroTol )
            {
                // TODO: Add while loop?
                F chi = QRBuf[(k-1)+k*QRLDim]/rho_km1_km1;
                if( Abs(RealPart(chi)) > ctrl.eta ||
                    Abs(ImagPart(chi)) > ctrl.eta )
                {
                    chi = Round(chi);
                    blas::Axpy
                    ( k, -chi,
                      &QRBuf[(k-1)*QRLDim], 1,
                      &QRBuf[ k   *QRLDim], 1 );

                    blas::Axpy
                    ( m, -Z(chi),
                      &BBuf[(k-1)*BLDim], 1,
                      &BBuf[ k   *BLDim], 1 );

                    if( formU )
                        blas::Axpy
                        ( n, -Z(chi),
                          &UBuf[(k-1)*ULDim], 1,
                          &UBuf[ k   *ULDim], 1 );

                    colUpdated = true;
                }
            }
        }
        else
        {
            vector<F> xBuf(k);
            // NOTE: Unless LLL is being aggressively executed in low precision,
            //       this loop should only need to be executed once
            const Int maxSizeReductions = 128;
            for( Int reduce=0; reduce<maxSizeReductions; ++reduce )
            {
                Int numNonzero = 0;
                for( Int i=k-1; i>=0; --i )
                {
                    F chi = QRBuf[i+k*QRLDim]/QRBuf[i+i*QRLDim];
                    if( Abs(RealPart(chi)) > ctrl.eta ||
                        Abs(ImagPart(chi)) > ctrl.eta )
                    {
                        chi = Round(chi);
                        blas::Axpy
                        ( i+1, -chi,
                          &QRBuf[i*QRLDim], 1,
                          &QRBuf[k*QRLDim], 1 );
                        ++numNonzero;
                    }
                    else
                        chi = 0;
                    xBuf[i] = chi;
                }
                if( numNonzero == 0 )
                    break;

                colUpdated = true;

                const float nonzeroRatio = float(numNonzero)/float(k); 
                if( nonzeroRatio >= ctrl.blockingThresh && k >= ctrl.minColThresh )
                {
                    vector<Z> xzBuf(k);
                    // Need array of type Z
                    for( Int i=0; i<k; ++i)
                        xzBuf[i] = Z(xBuf[i]);
                
                    blas::Gemv
                    ( 'N', m, k,
                      Z(-1), &BBuf[0*BLDim], BLDim,
                             &xzBuf[0],       1,
                      Z(+1), &BBuf[k*BLDim], 1 );
                    if( formU )
                        blas::Gemv
                        ( 'N', n, k,
                          Z(-1), &UBuf[0*ULDim], ULDim,
                                 &xzBuf[0],       1,
                          Z(+1), &UBuf[k*ULDim], 1 );
                }
                else
                {
                    for( Int i=k-1; i>=0; --i )
                    {
                        const Z chi = Z(xBuf[i]);
                        if( chi == Z(0) )
                            continue;
                        blas::Axpy
                        ( m, -chi,
                          &BBuf[i*BLDim], 1,
                          &BBuf[k*BLDim], 1 );
                        if( formU )
                            blas::Axpy
                            ( n, -chi,
                              &UBuf[i*ULDim], 1,
                              &UBuf[k*ULDim], 1 );
                    }
                }
            }
        }

        if ( !colUpdated )
        {
            break;
        }

        colUpdated = false;

        Real newNorm = lll::Norm2<Z,F>(B, bcol, k);
        auto rCol  = QR( ALL, IR(k) );
        Real rNorm = El::FrobeniusNorm(rCol);

        if (Abs(newNorm - rNorm)/newNorm >= thresh)
        {
            if ( ctrl.progress )
                Output("Repeating size reduction with k=", k, 
                       " because ||bk||=", newNorm, ", ||rk||=", rNorm);
            continue;
        }

        if( ctrl.time )
            roundTimer.Stop();
        if( !ctrl.unsafeSizeReduct && !limits::IsFinite(newNorm) )
            RuntimeError("Encountered an unbounded norm; increase precision");
        if( !ctrl.unsafeSizeReduct && newNorm > Real(1)/eps )
            RuntimeError("Encountered norm greater than 1/eps, where eps=",eps);

        
        if( newNorm > ctrl.reorthogTol*colNorms.Get(k,0) )
        {
            break;
        }
        else if( ctrl.progress )
            Output
            ("  Reorthogonalizing with k=",k,
             " since oldNorm=",colNorms.Get(k,0)," and newNorm=",newNorm);

        colNorms.Set(k,0, newNorm);
    }
    lll::HouseholderStep( k, QR, t, d, ctrl.time );
    if( ctrl.time )
        stepTimer.Stop();
    return false;
}

// Consider explicitly returning both Q and R rather than just R (in 'QR')
template<typename Z, typename F>
LLLInfo<Base<F>> LeftAlg
( Matrix<Z>& B,
  Matrix<Z>& U,
  Matrix<F>& QR,
  Matrix<F>& t,
  Matrix<Base<F>>& d,
  bool formU,
  const LLLCtrl<Base<F>>& ctrl )
{
    DEBUG_ONLY(CSE cse("lll::LeftAlg"))
    typedef Base<F> Real;
    if( ctrl.time )
    {
        stepTimer.Reset();
        houseStepTimer.Reset();
        houseViewTimer.Reset();
        houseReflectTimer.Reset();
        applyHouseTimer.Reset();
        roundTimer.Reset();
    }

    const Int m = B.Height();
    const Int n = B.Width();
    const Int minDim = Min(m,n);

    // Keep this vector around for norm computation purposes
    // Avoid repeatedly reallocating memory
    Matrix<Real> bcol;
    Zeros(bcol, m, 1);

    Matrix<Real> colNorms;
    Zeros( colNorms, n, 1 );

    for (Int i=0; i<n; i++)
    {
        auto col = B( ALL, IR(i) );
        Copy(col, bcol);
        colNorms.Set( i, 0, El::FrobeniusNorm(bcol) );
    }

    Int numSwaps=0;
    Int nullity = 0;
    Int firstSwap = n;
    if( ctrl.jumpstart && ctrl.startCol > 0 )
    {
        if( QR.Height() != m || QR.Width() != n )
            LogicError
            ("QR was ",QR.Height()," x ",QR.Width()," and should have been ",
             m," x ",n);
        if( t.Height() != minDim || t.Width() != 1 )
            LogicError
            ("t was ",t.Height(),", x ",t.Width()," and should have been ",
             "Min(m,n)=Min(",m,",",n,")=",Min(m,n)," x 1");
        if( d.Height() != minDim || d.Width() != 1 )
            LogicError
            ("d was ",d.Height(),", x ",d.Width()," and should have been ",
             "Min(m,n)=Min(",m,",",n,")=",Min(m,n)," x 1");
    }
    else
    {
        Zeros( QR, m, n );
        Zeros( t, minDim, 1 );
        Zeros( d, minDim, 1 );
        while( true )
        {
            // Perform the first step of Householder reduction
            lll::ExpandQR( 0, B, QR, t, d, ctrl.numOrthog, ctrl.time );
            lll::HouseholderStep( 0, QR, t, d, ctrl.time );
            if( QR.GetRealPart(0,0) <= ctrl.zeroTol )
            {
                auto b0 = B(ALL,IR(0));
                auto QR0 = QR(ALL,IR(0));
                Zero( b0 );
                Zero( QR0 );
                t.Set( 0, 0, Real(2) );
                d.Set( 0, 0, Real(1) );

                ColSwap( B, 0, (n-1)-nullity );
                if( formU )
                    ColSwap( U, 0, (n-1)-nullity );

                // Swap the column norms
                Base<F> tmp = colNorms.Get((n-1)-nullity, 0);
                colNorms.Set( (n-1)-nullity, 0, colNorms.Get(0, 0) );
                colNorms.Set( 0, 0, tmp );

                ++nullity;
                ++numSwaps;
                firstSwap = 0;
            }
            else
                break;
            if( nullity >= n )
                break;
        }
    }

    Int k = ( ctrl.jumpstart ? Max(ctrl.startCol,1) : 1 );
    while( k < n-nullity )
    {
        bool zeroVector = lll::Step( k, B, U, QR, t, d, bcol, colNorms, formU, ctrl );
        if( zeroVector )
        {
            ColSwap( B, k, (n-1)-nullity );
            if( formU )
                ColSwap( U, k, (n-1)-nullity );
            ++nullity;
            ++numSwaps;
            firstSwap = Min(firstSwap,k);
            continue;
        }

        const Real rho_km1_km1 = QR.GetRealPart(k-1,k-1);
        const F rho_km1_k = QR.Get(k-1,k);
        const Real rho_k_k = ( k >= m ? Real(0) : QR.GetRealPart(k,k) ); 
        
        const Real leftTerm = Sqrt(ctrl.delta)*rho_km1_km1;
        const Real rightTerm = lapack::SafeNorm(rho_k_k,rho_km1_k);
        // NOTE: It is possible that, if delta < 1/2, that rho_k_k could be
        //       zero and the usual Lovasz condition would be satisifed.
        //       For this reason, we explicitly force a pivot if R(k,k) is
        //       deemed to be numerically zero.
        if( leftTerm <= rightTerm && rho_k_k > ctrl.zeroTol )
        {
            ++k;
        }
        else
        {
            ++numSwaps;
            firstSwap = Min(firstSwap,k-1);
            if( ctrl.progress )
            {
                if( rho_k_k <= ctrl.zeroTol )
                    Output("Dropping from k=",k," because R(k,k) ~= 0");
                else
                    Output
                    ("Dropping from k=",k," to ",Max(k-1,1),
                     " since sqrt(delta)*R(k-1,k-1)=",leftTerm," > ",rightTerm);
            }

            ColSwap( B, k-1, k );
            if( formU )
                ColSwap( U, k-1, k );
				
			// Swap the column norms
			Base<F> tmp = colNorms.Get(k, 0);
            colNorms.Set( k, 0, colNorms.Get(k-1, 0) );
            colNorms.Set( k-1, 0, tmp );

            if( k == 1 )
            {
                while( true )
                {
                    // We must reinitialize since we keep k=1
                    lll::ExpandQR( 0, B, QR, t, d, ctrl.numOrthog, ctrl.time );
                    lll::HouseholderStep( 0, QR, t, d, ctrl.time );
                    if( QR.GetRealPart(0,0) <= ctrl.zeroTol )
                    {
                        auto b0 = B(ALL,IR(0));
                        auto QR0 = QR(ALL,IR(0));
                        Zero( b0 );
                        Zero( QR0 );
                        t.Set( 0, 0, Real(2) );
                        d.Set( 0, 0, Real(1) );

                        ColSwap( B, 0, (n-1)-nullity );
                        if( formU )
                            ColSwap( U, 0, (n-1)-nullity );

                        // Swap the column norms
                        Base<F> tmp = colNorms.Get((n-1)-nullity, 0);
                        colNorms.Set( (n-1)-nullity, 0, colNorms.Get(0, 0) );
                        colNorms.Set( 0, 0, tmp );
							
                        ++nullity;
                        ++numSwaps;
                        firstSwap = 0;
                    }
                    else
                        break;
                    if( nullity >= n )
                        break;
                }
            }
            else
            {
                k = k-1; 
            }
        }
    }

    if( ctrl.time )
    {
        Output("  Step time:                ",stepTimer.Total());
        Output("    Householder step time:  ",houseStepTimer.Total());
        Output("      view time:            ",houseViewTimer.Total());
        Output("      reflect time:         ",houseReflectTimer.Total());
        Output("    Apply Householder time: ",applyHouseTimer.Total());
        Output("    Round time:             ",roundTimer.Total());
    }

    std::pair<Real,Real> achieved = lll::Achieved(QR,ctrl);
    Real logVol = lll::LogVolume(QR);

    LLLInfo<Base<F>> info;
    info.delta = achieved.first;
    info.eta = achieved.second;
    info.rank = n-nullity;
    info.nullity = nullity;
    info.numSwaps = numSwaps;
    info.firstSwap = firstSwap;
    info.logVol = logVol;

    return info;
}

template<typename Z, typename F>
LLLInfo<Base<F>> LeftDeepAlg
( Matrix<Z>& B,
  Matrix<Z>& U,
  Matrix<F>& QR,
  Matrix<F>& t,
  Matrix<Base<F>>& d,
  bool formU,
  const LLLCtrl<Base<F>>& ctrl )
{
    DEBUG_ONLY(CSE cse("lll::LeftDeepAlg"))
    typedef Base<F> Real;
    if( ctrl.delta <= Real(1)/Real(2) )
        LogicError
        ("Deep insertion requires delta > 1/2 for handling dependence");
    if( ctrl.time )
    {
        stepTimer.Reset();
        houseStepTimer.Reset();
        houseViewTimer.Reset();
        houseReflectTimer.Reset();
        applyHouseTimer.Reset();
        roundTimer.Reset();
    }

    const Int m = B.Height();
    const Int n = B.Width();
    const Int minDim = Min(m,n);

	// Keep this vector around for norm computation purposes
    // Avoid repeatedly reallocating memory
    Matrix<Real> bcol;
    Zeros(bcol, m, 1);

	Matrix<Real> colNorms;
    Zeros( colNorms, n, 1 );

    for (Int i=0; i<n; i++)
    {
        auto col = B( ALL, IR(i) );
        Copy(col, bcol);
        colNorms.Set( i, 0, El::FrobeniusNorm(bcol) );
    }
	
    // TODO: Move into a control structure
    const bool alwaysRecomputeNorms = false;
    const Real updateTol = Sqrt(limits::Epsilon<Real>());

    Int numSwaps=0;
    Int nullity = 0;
    Int firstSwap = n;
    if( ctrl.jumpstart && ctrl.startCol > 0 )
    {
        if( QR.Height() != m || QR.Width() != n )
            LogicError
            ("QR was ",QR.Height()," x ",QR.Width()," and should have been ",
             m," x ",n);
        if( t.Height() != minDim || t.Width() != 1 )
            LogicError
            ("t was ",t.Height(),", x ",t.Width()," and should have been ",
             "Min(m,n)=Min(",m,",",n,")=",Min(m,n)," x 1");
        if( d.Height() != minDim || d.Width() != 1 )
            LogicError
            ("d was ",d.Height(),", x ",d.Width()," and should have been ",
             "Min(m,n)=Min(",m,",",n,")=",Min(m,n)," x 1");
    }
    else
    {
        Zeros( QR, m, n );
        Zeros( d, minDim, 1 );
        Zeros( t, minDim, 1 );

        while( true )
        {   
            // Perform the first step of Householder reduction
            lll::ExpandQR( 0, B, QR, t, d, ctrl.numOrthog, ctrl.time );
            lll::HouseholderStep( 0, QR, t, d, ctrl.time );
            if( QR.GetRealPart(0,0) <= ctrl.zeroTol )
            {
                auto b0 = B(ALL,IR(0));
                auto QR0 = QR(ALL,IR(0));
                Zero( b0 );
                Zero( QR0 ); 
                t.Set( 0, 0, Real(2) );
                d.Set( 0, 0, Real(1) );

                ColSwap( B, 0, (n-1)-nullity );
                if( formU )
                    ColSwap( U, 0, (n-1)-nullity );

                // Swap the column norms
                Base<F> tmp = colNorms.Get((n-1)-nullity, 0);
                colNorms.Set( (n-1)-nullity, 0, colNorms.Get(0, 0) );
                colNorms.Set( 0, 0, tmp );
					
                ++nullity;
                ++numSwaps;
                firstSwap = 0;
            }
            else
                break;
            if( nullity >= n )
                break;
        }
    }

    Int k = ( ctrl.jumpstart ? Max(ctrl.startCol,1) : 1 );
    while( k < n-nullity )
    {
        bool zeroVector = lll::Step( k, B, U, QR, t, d, bcol, colNorms, formU, ctrl );
        if( zeroVector )
        {
            ColSwap( B, k, (n-1)-nullity );
            if( formU )
                ColSwap( U, k, (n-1)-nullity );
            ++nullity;
            ++numSwaps;
            firstSwap = Min(firstSwap,k);
            continue;
        }

        bool swapped=false;
        // NOTE:
        // There appears to be a mistake in the "New Step 4" initialization of 
        // "c" in 
        //
        //   Schnorr and Euchner, "Lattice Basis Reduction: Improved Practical
        //   Algorithms and Solving Subset Sum Problems", 
        //
        // as "c" should be initialized to || b_k ||^2, not || b'_k ||^2,
        // where || b'_k ||_2 = R(k,k) and || b_k ||_2 = norm(R(1:k,k)),
        // if we count from one.
        const Int rColHeight = Min(k+1,minDim);
        Real origNorm = blas::Nrm2( rColHeight, QR.LockedBuffer(0,k), 1 );
        Real partialNorm = origNorm;
        for( Int i=0; i<Min(k,minDim); ++i )
        {
            const Real rho_i_i = QR.GetRealPart(i,i);
            const Real leftTerm = Sqrt(ctrl.delta)*rho_i_i;
            if( leftTerm > partialNorm )
            {
                ++numSwaps;
                firstSwap = Min(firstSwap,i);
                if( ctrl.progress )
                    Output("Deep inserting k=",k," into position i=",i," since sqrt(delta)*R(i,i)=",leftTerm," > ",partialNorm);

                DeepColSwap( B, i, k );
                if( formU )
                    DeepColSwap( U, i, k );

                // Todo: Check that this is actually correct behaviour
                // Swap the column norms
                Base<F> tmp = colNorms.Get(k, 0);
                colNorms.Set( k, 0, colNorms.Get(i, 0) );
                colNorms.Set( i, 0, tmp );
				
                if( i == 0 )
                {
                    while( true )
                    {
                        // We must reinitialize since we keep k=1
                        lll::ExpandQR
                        ( 0, B, QR, t, d, ctrl.numOrthog, ctrl.time );
                        lll::HouseholderStep( 0, QR, t, d, ctrl.time );
                        if( QR.GetRealPart(0,0) <= ctrl.zeroTol )
                        {
                            auto b0 = B(ALL,IR(0));
                            auto QR0 = QR(ALL,IR(0));
                            Zero( b0 );
                            Zero( QR0 );
                            t.Set( 0, 0, Real(2) );
                            d.Set( 0, 0, Real(1) );

                            ColSwap( B, 0, (n-1)-nullity );
                            if( formU )
                                ColSwap( U, 0, (n-1)-nullity );

                            // Swap the column norms
                            Base<F> tmp = colNorms.Get((n-1)-nullity, 0);
                            colNorms.Set( (n-1)-nullity, 0, colNorms.Get(0, 0) );
                            colNorms.Set( 0, 0, tmp );								

                            ++nullity;
                            ++numSwaps;
                            firstSwap = 0;
                        }
                        else
                            break;
                        if( nullity >= n )
                            break;
                    }
                    k=1;
                }
                else
                {
                    k = i;
                }
                swapped = true;
                break;
            }
            else
            {
                // Downdate the partial norm in the same manner as LAWN 176
                Real gamma = Abs(QR.Get(i,k)) / partialNorm;
                gamma = Max( Real(0), (Real(1)-gamma)*(Real(1)+gamma) );
                const Real ratio = partialNorm / origNorm; 
                const Real phi = gamma*(ratio*ratio);
                if( phi <= updateTol || alwaysRecomputeNorms )
                {
                    partialNorm = blas::Nrm2
                    ( rColHeight-(i+1), QR.LockedBuffer(i+1,k), 1 );
                    origNorm = partialNorm;
                }
                else
                    partialNorm *= Sqrt(gamma);
            }
        }
        if( !swapped )
            ++k;
    }

    if( ctrl.time )
    {
        Output("  Step time:              ",stepTimer.Total());
        Output("    Householder step time:  ",houseStepTimer.Total());
        Output("      view time:              ",houseViewTimer.Total());
        Output("      reflect time:           ",houseReflectTimer.Total());
        Output("    Apply Householder time: ",applyHouseTimer.Total());
        Output("    Round time:             ",roundTimer.Total());
    }

    std::pair<Real,Real> achieved = lll::Achieved(QR,ctrl);
    Real logVol = lll::LogVolume(QR);

    LLLInfo<Base<F>> info;
    info.delta = achieved.first;
    info.eta = achieved.second;
    info.rank = n-nullity;
    info.nullity = nullity;
    info.numSwaps = numSwaps;
    info.firstSwap = firstSwap;
    info.logVol = logVol;

    return info;
}

template<typename Z, typename F>
LLLInfo<Base<F>> LeftDeepReduceAlg
( Matrix<Z>& B,
  Matrix<Z>& U,
  Matrix<F>& QR,
  Matrix<F>& t,
  Matrix<Base<F>>& d,
  bool formU,
  const LLLCtrl<Base<F>>& ctrl )
{
    DEBUG_ONLY(CSE cse("lll::LeftDeepReduceAlg"))
    typedef Base<F> Real;
    if( ctrl.delta <= Real(1)/Real(2) )
        LogicError
        ("Deep insertion requires delta > 1/2 for handling dependence");
    if( ctrl.time )
    {
        stepTimer.Reset();
        houseStepTimer.Reset();
        houseViewTimer.Reset();
        houseReflectTimer.Reset();
        applyHouseTimer.Reset();
        roundTimer.Reset();
    }

    const Int m = B.Height();
    const Int n = B.Width();
    const Int minDim = Min(m,n);

	// Keep this vector around for norm computation purposes
    // Avoid repeatedly reallocating memory
    Matrix<Real> bcol;
    Zeros(bcol, m, 1);

	Matrix<Real> colNorms;
    Zeros( colNorms, n, 1 );

    for (Int i=0; i<n; i++)
    {
        auto col = B( ALL, IR(i) );
        Copy(col, bcol);
        colNorms.Set( i, 0, El::FrobeniusNorm(bcol) );
    }	

    Int numSwaps = 0;
    Int nullity = 0;
    Int firstSwap = n;
    if( ctrl.jumpstart && ctrl.startCol > 0 )
    {
        if( QR.Height() != m || QR.Width() != n )
            LogicError
            ("QR was ",QR.Height()," x ",QR.Width()," and should have been ",
             m," x ",n);
        if( t.Height() != minDim || t.Width() != 1 )
            LogicError
            ("t was ",t.Height(),", x ",t.Width()," and should have been ",
             "Min(m,n)=Min(",m,",",n,")=",Min(m,n)," x 1");
        if( d.Height() != minDim || d.Width() != 1 )
            LogicError
            ("d was ",d.Height(),", x ",d.Width()," and should have been ",
             "Min(m,n)=Min(",m,",",n,")=",Min(m,n)," x 1");
    }
    else
    {
        Zeros( QR, m, n );
        Zeros( d, minDim, 1 );
        Zeros( t, minDim, 1 );

        while( true )
        {
            // Perform the first step of Householder reduction
            lll::ExpandQR( 0, B, QR, t, d, ctrl.numOrthog, ctrl.time );
            lll::HouseholderStep( 0, QR, t, d, ctrl.time );
            if( QR.GetRealPart(0,0) <= ctrl.zeroTol )
            {
                auto b0 = B(ALL,IR(0));
                auto QR0 = QR(ALL,IR(0));
                Zero( b0 );
                Zero( QR0 );
                t.Set( 0, 0, Real(2) );
                d.Set( 0, 0, Real(1) );

                ColSwap( B, 0, (n-1)-nullity );
                if( formU )
                    ColSwap( U, 0, (n-1)-nullity );

                // Swap the column norms
                Base<F> tmp = colNorms.Get((n-1)-nullity, 0);
                colNorms.Set( (n-1)-nullity, 0, colNorms.Get(0, 0) );
                colNorms.Set( 0, 0, tmp );					

                ++nullity;
                ++numSwaps;
                firstSwap = 0;
            }
            else
                break;
            if( nullity >= n )
                break;
        }
    }

    Int k = ( ctrl.jumpstart ? Max(ctrl.startCol,1) : 1 );
    while( k < n-nullity )
    {
        bool zeroVector = lll::Step( k, B, U, QR, t, d, bcol, colNorms, formU, ctrl );
        if( zeroVector )
        {
            ColSwap( B, k, (n-1)-nullity );
            if( formU )
                ColSwap( U, k, (n-1)-nullity );
            ++nullity;
            ++numSwaps;
            firstSwap = Min(firstSwap,k);
            continue;
        }

        bool swapped=false;
        const Int rColHeight = Min(k+1,minDim);
        for( Int i=0; i<Min(k,minDim); ++i )
        {
            // Perform additional reduction before attempting deep insertion
            // and reverse them if the candidate was not chosen 
            // (otherwise |R(i,j)|/R(i,i) can be greater than 1/2 for 
            // some j > i)
            // TODO: Add a while loop version for low-precision reduction
            auto rk = QR( IR(0,rColHeight), IR(k) );
            auto rkCopy( rk );
            bool deepReduced = false;
            Matrix<F> x;
            Zeros( x, Min(k,minDim)-i, 1 );
            for( Int l=i; l<Min(k,minDim); ++l )
            {
                // TODO: Perform this calculation more carefully, perhaps
                //       with an equivalent of the scaled squaring approach
                //       used for norms
                F dot = blas::Dot(l-i+1,QR.Buffer(i,k),1,QR.Buffer(i,l),1);
                Real nrm = blas::Nrm2(l-i+1,QR.Buffer(i,l),1);
                F mu = (dot/nrm)/nrm;
                if( ctrl.delta*Abs(RealPart(mu)) >= Real(1)/Real(2) ||
                    ctrl.delta*Abs(ImagPart(mu)) >= Real(1)/Real(2) )
                {
                    F chi = Round(mu);
                    x.Set( l-i, 0, chi );
                    blas::Axpy
                    ( l+1, -chi,
                      QR.Buffer(0,l), 1,
                      QR.Buffer(0,k), 1 );
                    deepReduced = true;
                }
            }

            const Real rho_i_i = QR.GetRealPart(i,i);
            const Real leftTerm = Sqrt(ctrl.delta)*rho_i_i;
            const Real partialNorm =
              blas::Nrm2( rColHeight-i, QR.LockedBuffer(i,k), 1 );
            if( leftTerm > partialNorm )
            {
                ++numSwaps;
                firstSwap = Min(firstSwap,i);
                if( ctrl.progress )
                    Output("Deep inserting k=",k," into position i=",i," since sqrt(delta)*R(i,i)=",leftTerm," > ",partialNorm);

                // Finish applying the deep reductions since they were accepted
                // TODO: Apply these in a batch instead?
                for( Int l=i; l<Min(k,minDim); ++l )
                {
                    F chi = x.Get(l-i,0);
                    if( Abs(RealPart(chi)) > 0 || Abs(ImagPart(chi)) > 0 )
                    {
                        blas::Axpy
                        ( m, -Z(chi),
                          B.Buffer(0,l), 1,
                          B.Buffer(0,k), 1 );
                        if( formU )
                            blas::Axpy
                            ( n, -Z(chi),
                              U.Buffer(0,l), 1,
                              U.Buffer(0,k), 1 );
                    }
                }

                DeepColSwap( B, i, k );
                if( formU )
                    DeepColSwap( U, i, k );

                // Update the column norms
                colNorms.Set( k, 0, colNorms.Get(i, 0) );
                colNorms.Set( i, 0, lll::Norm2(B, bcol, i) );						
				
                if( i == 0 )
                {
                    while( true )
                    {
                        // We must reinitialize since we keep k=1
                        lll::ExpandQR
                        ( 0, B, QR, t, d, ctrl.numOrthog, ctrl.time );
                        lll::HouseholderStep( 0, QR, t, d, ctrl.time );
                        if( QR.GetRealPart(0,0) <= ctrl.zeroTol )
                        {
                            auto b0 = B(ALL,IR(0));
                            auto QR0 = QR(ALL,IR(0));
                            Zero( b0 );
                            Zero( QR0 );
                            t.Set( 0, 0, Real(2) );
                            d.Set( 0, 0, Real(1) );

                            ColSwap( B, 0, (n-1)-nullity );
                            if( formU )
                                ColSwap( U, 0, (n-1)-nullity );
                       
                            ++nullity;
                            ++numSwaps;
                            firstSwap = 0;
                        }
                        else
                            break;
                        if( nullity >= n )
                            break;
                    }
                    k=1;
                }
                else
                {
                    k = i;
                }
                swapped = true;
                break;
            }
            else if( deepReduced )
            {
                // Undo the (partially applied) deep reductions
                rk = rkCopy;
            }
        }
        if( !swapped )
            ++k;
    }

    if( ctrl.time )
    {
        Output("  Step time:              ",stepTimer.Total());
        Output("    Householder step time:  ",houseStepTimer.Total());
        Output("      view time:              ",houseViewTimer.Total());
        Output("      reflect time:           ",houseReflectTimer.Total());
        Output("    Apply Householder time: ",applyHouseTimer.Total());
        Output("    Round time:             ",roundTimer.Total());
    }

    std::pair<Real,Real> achieved = lll::Achieved(QR,ctrl);
    Real logVol = lll::LogVolume(QR);

    LLLInfo<Base<F>> info;
    info.delta = achieved.first;
    info.eta = achieved.second;
    info.rank = n-nullity;
    info.nullity = nullity;
    info.numSwaps = numSwaps;
    info.firstSwap = firstSwap;
    info.logVol = logVol;

    return info;
}

} // namespace lll
} // namespace El

#endif // ifndef EL_LATTICE_LLL_LEFT_HPP
