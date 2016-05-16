/*
   Copyright (c) 2009-2016, Jack Poulson, 2016, Ron Estrin
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_LATTICE_LLL_LEFT_HPP
#define EL_LATTICE_LLL_LEFT_HPP

namespace El {
namespace lll {

// This struct is used as a 'global' variable
// to avoid reallocating memory for the 'local'
// variables inside.
template<typename Z, typename F>
struct SharedNorms
{
	Matrix<Base<F>> colNorms;
	Matrix<Int> colExpo;
	Matrix<F> bcol;
	Matrix<Z> bzcol;
};

// Put the k'th column of B into the k'th column of QR and then rotate
// said column with the first k-1 (scaled) Householder reflectors.

template<typename Z, typename F>
void ExpandQR
( Int k,
  const Matrix<Z>& B,
        Matrix<F>& QR,
  const Matrix<F>& t,
  const Matrix<Base<F>>& d,
  const SharedNorms<Z,F>& sharedNorms,
  Int numOrthog,
  bool colExpo,
  bool time )
{
    DEBUG_ONLY(CSE cse("lll::ExpandQR"))
    typedef Base<F> Real;
    const Int m = B.Height();
    const Int n = B.Width();
    const Int minDim = Min(m,n);

    // Copy in the k'th column of B
	if( colExpo )
	{
		Real scale = Pow(F(2), F(sharedNorms.colExpo(k,0)));
		for( Int i=0; i<m; ++i )
			QR(i,k) = F(B(i,k)) / scale;
	}
	else
	{
		for( Int i=0; i<m; ++i )
			QR(i,k) = F(B(i,k));
	}

    if( k == 0 )
        return;

    if( time )
        applyHouseTimer.Start();
    for( Int orthog=0; orthog<numOrthog; ++orthog )
    {
        if ( k < 0)
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
                // Apply the i'th Householder reflectors
 
                // Temporarily replace QR(i,i) with 1
                const Real alpha = RealPart(QR(i,i));
                QR(i,i) = 1;

                const F innerProd =
                    blas::Dot
                    ( m-i,
                      &QR(i,i), 1,
                      &QR(i,k), 1 );
                    blas::Axpy
                    ( m-i, -t(i)*innerProd,
                      &QR(i,i), 1,
                      &QR(i,k), 1 );

                    // Fix the scaling
                    QR(i,k) *= d(i);

                    // Restore H(i,i)
                    QR(i,i) = alpha;
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
    F& rhokk = QR(k,k); 
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
    t(k) = tau;
    if( RealPart(rhokk) < Real(0) )
    {
        d(k) = -1;
        rhokk *= -1;
    }
    else
        d(k) = +1;

    if( time )
        houseStepTimer.Stop();
}

template<typename Z, typename F>
Base<F> Norm2
( Matrix<Z>& B,
  SharedNorms<Z,F>& sharedNorms,
  Int k,
  bool colExpo,
  bool time )
{
    if( time )
        normTimer.Start();
	if( colExpo )
	{
		auto col = B( ALL, IR(k) );
		Base<Z> maxNorm = El::MaxNorm( col );
		Copy( col, sharedNorms.bzcol );
		Int expo = Int(Ceil(Log2(maxNorm)));
		sharedNorms.bzcol *= Pow(Z(2),Z(-expo));
		Copy(sharedNorms.bzcol, sharedNorms.bcol);
		sharedNorms.colExpo(k, 0) = expo;
	}
	else
	{
		auto col = B(ALL, IR(k));
		Copy(col, sharedNorms.bcol);
	}
	Base<F> norm = El::FrobeniusNorm(sharedNorms.bcol);
    if( time )
        normTimer.Stop();
    return norm;
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
  bool formU,
  SharedNorms<Z,F>& sharedNorms,
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

    bool colUpdated = false;

    while( true )
    {
	    Real oldNorm = sharedNorms.colNorms.Get(k,0);
        if( !ctrl.unsafeSizeReduct && !limits::IsFinite(oldNorm) )
            RuntimeError("Encountered an unbounded norm; increase precision");
        if( !ctrl.unsafeSizeReduct && oldNorm > Real(1)/eps )
            RuntimeError("Encountered norm greater than 1/eps, where eps=",eps);

        if( oldNorm <= ctrl.zeroTol ) //  Adjust for scale
        {
            for( Int i=0; i<m; ++i )
                B(i,k) = 0;
            for( Int i=0; i<m; ++i )
                QR(i,k) = 0;
            if( k < Min(m,n) )
            {
                t(k) = Real(2);
                d(k) = Real(1);
            }
            if( ctrl.time )
                stepTimer.Stop();
            return true;
        }

		//Print(sharedNorms.colExpo, "colExpo");
		//Print(QR, "QR before expandqr");
        lll::ExpandQR( k, B, QR, t, d, sharedNorms, ctrl.numOrthog, ctrl.colExpo, ctrl.time );
		//Print(sharedNorms.colExpo, "colExpo");
		//Print(QR, "QR after expandqr");
		
        if( ctrl.time )
            roundTimer.Start();

        if( ctrl.variant == LLL_WEAK )
        {
            const Real rho_km1_km1 = RealPart(QR(k-1,k-1));
            if( rho_km1_km1 > ctrl.zeroTol ) // TODO: Adjust for scale
            {
                // TODO: Add while loop?
                F chi = QR(k-1,k)/rho_km1_km1;
				Real scale = 1;
				if( ctrl.colExpo )
				{
					Int expo = (sharedNorms.colExpo(k-1,0) - sharedNorms.colExpo(k,0));
					scale = Pow(F(2), F(-expo));
				}
                if( Abs(RealPart(chi*scale)) > ctrl.eta ||
                    Abs(ImagPart(chi*scale)) > ctrl.eta )
                {
                    chi = Round(chi);
                    blas::Axpy
                    ( k, -chi,
                      &QR(0,k-1), 1,
                      &QR(0,k  ), 1 );

                    blas::Axpy
                    ( m, Z(-chi*scale),
                      &B(0,k-1), 1,
                      &B(0,k  ), 1 );

                    if( formU )
                        blas::Axpy
                        ( n, Z(-chi*scale),
                          &U(0,k-1), 1,
                          &U(0,k  ), 1 );

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
			    //Print(QR, "QR pre reduction");
			
                Int numNonzero = 0;
                for( Int i=k-1; i>=0; --i )
                {
                    F chi = QR(i,k)/QR(i,i);
					Real scale = 1;
					if( ctrl.colExpo )
					{
						Int expo = (sharedNorms.colExpo(i,0) - sharedNorms.colExpo(k,0));
						scale = Pow(F(2), F(-expo));
					}
					//cout << "ii = " << i << endl;
					//cout << "chi = " << chi*scale << endl;
                    if( Abs(RealPart(chi*scale)) > ctrl.eta ||
                        Abs(ImagPart(chi*scale)) > ctrl.eta )
                    {
						chi = Round(chi*scale)/scale;
						blas::Axpy
						( i+1, -chi,
						  &QR(0,i), 1,
						  &QR(0,k), 1 );
                        ++numNonzero;
                    }
                    else
                        chi = 0;
                    xBuf[i] = scale*chi;
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
                      Z(-1), &B(0,0),  B.LDim(),
                             &xzBuf[0], 1,
                      Z(+1), &B(0,k),  1 );
                    if( formU )
                        blas::Gemv
                        ( 'N', n, k,
                          Z(-1), &U(0,0),  U.LDim(),
                                 &xzBuf[0], 1,
                          Z(+1), &U(0,k),  1 );
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
                          &B(0,i), 1,
                          &B(0,k), 1 );
                        if( formU )
                            blas::Axpy
                            ( n, -chi,
                              &U(0,i), 1,
                              &U(0,k), 1 );
                    }
                }
            }
        }
        if( ctrl.time )
            roundTimer.Stop();
 
        if( !colUpdated )
        {
            break;
        }

        colUpdated = false;

//		//Print(QR, "QR after size reduction");
		
		Int oldExpo = 0;
		if( ctrl.colExpo )
			oldExpo = sharedNorms.colExpo(k,0);
		
        Real newNorm = lll::Norm2<Z,F>(B, sharedNorms, k, ctrl.colExpo, ctrl.time);

		F scale = 1;
		if( ctrl.colExpo )
		{
//			cout << "k = " << k << endl;
//			Print( sharedNorms.colExpo, "colExpo" );
//			cout << "oldExpo = " << oldExpo << endl;
//			cout << "newNorm = " << newNorm << endl;
//			auto bcol = B(ALL,IR(k));
//			cout << "||b_k|| = " << El::FrobeniusNorm(bcol) << endl;
			
			scale = Pow(F(2), F(-sharedNorms.colExpo(k,0) + oldExpo));
		}

//		//cout << "newNorm = " << newNorm << endl;

		auto rCol  = QR( ALL, IR(k) );
		rCol *= scale;
        if( ctrl.time )
            normTimer.Start();
        Real rNorm = El::FrobeniusNorm(rCol);
        if( ctrl.time )
            normTimer.Stop();

//		cout << "||r_k|| = " << rNorm << endl;
//		cout << "||r_k||scaled = " << rNorm*Pow(F(2), F(sharedNorms.colExpo(k,0))) << endl;
//		cout << endl;
			
		Real error = Abs(newNorm - rNorm)/newNorm;

        if( error >= thresh )
        {
            if( ctrl.progress )
                Output("Repeating size reduction with k=", k, 
                       " because ||bk||=", newNorm, ", ||rk||=", rNorm);
            continue;
        }

        if( !ctrl.unsafeSizeReduct && !limits::IsFinite(newNorm) )
            RuntimeError("Encountered an unbounded norm; increase precision");
        if( !ctrl.unsafeSizeReduct && newNorm > Real(1)/eps )
            RuntimeError("Encountered norm greater than 1/eps, where eps=",eps);

        if( newNorm > ctrl.reorthogTol*oldNorm )
        {
            break;
        }
        else if( ctrl.progress )
            Output
            ("  Reorthogonalizing with k=",k,
             " since oldNorm=",sharedNorms.colNorms.Get(k,0)," and newNorm=",newNorm);

        sharedNorms.colNorms.Set(k,0, newNorm);
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
        normTimer.Reset();
    }

    const Int m = B.Height();
    const Int n = B.Width();
    const Int minDim = Min(m,n);

    // Keep this struct around for norm computation purposes
    // Avoid repeatedly reallocating memory
	SharedNorms<Z, F> sharedNorms;
    Zeros( sharedNorms.bcol, m, 1);
	Zeros( sharedNorms.bzcol, m, 1);
    Zeros( sharedNorms.colNorms, n, 1 );
	Zeros( sharedNorms.colExpo, n, 1 );
	
	if( ctrl.time )
        normTimer.Start();
	if( ctrl.colExpo )
	{
		for (Int i=0; i<n; i++)
		{
			auto col = B( ALL, IR(i) );
			Base<Z> maxNorm = El::MaxNorm( col );
			Copy( col, sharedNorms.bzcol );
			Int expo = Int(Ceil(Log2(maxNorm)));
			sharedNorms.bzcol *= Pow(Z(2),Z(-expo));
			Copy(sharedNorms.bzcol, sharedNorms.bcol);
			sharedNorms.colNorms(i, 0) = El::Nrm2(sharedNorms.bcol);
			sharedNorms.colExpo(i, 0) = expo;
		}
	}
	else
	{
		for (Int i=0; i<n; i++)
		{
			auto col = B( ALL, IR(i) );
			Copy(col, sharedNorms.bcol);
			sharedNorms.colNorms(i, 0) = El::Nrm2(sharedNorms.bcol);
		}
	}
    if( ctrl.time )
        normTimer.Stop();
	
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
            lll::ExpandQR( 0, B, QR, t, d, sharedNorms, ctrl.numOrthog, ctrl.colExpo, ctrl.time );
            lll::HouseholderStep( 0, QR, t, d, ctrl.time );
			Real scale = 1;
			if( ctrl.colExpo )
				scale = Pow(F(2), F(sharedNorms.colExpo(0,0)));
            if( RealPart(QR(0,0)) <= ctrl.zeroTol/scale )
            {
                auto b0 = B(ALL,IR(0));
                auto QR0 = QR(ALL,IR(0));
                Zero( b0 );
                Zero( QR0 );
                t(0) = Real(2);
                d(0) = Real(1);

                ColSwap( B, 0, (n-1)-nullity );
                if( formU )
                    ColSwap( U, 0, (n-1)-nullity );

                // Swap the column norms
                RowSwap( sharedNorms.colNorms, 0, (n-1)-nullity );
				if( ctrl.colExpo )
					RowSwap( sharedNorms.colExpo, 0, (n-1)-nullity );

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
		//Print(sharedNorms.colExpo, "colExpo");
		//Print(QR, "QR before step");
        bool zeroVector = lll::Step( k, B, U, QR, t, d, formU, sharedNorms, ctrl );
		//Print(sharedNorms.colExpo, "colExpo");		
		//Print(QR, "QR after step");
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

        const Real rho_km1_km1 = RealPart(QR(k-1,k-1));
        const F rho_km1_k = QR(k-1,k);
        const Real rho_k_k = ( k >= m ? Real(0) : RealPart(QR(k,k)) ); 
        
        const Real leftTerm = Sqrt(ctrl.delta)*rho_km1_km1;
        Real rightTerm = lapack::SafeNorm(rho_k_k,rho_km1_k);
		if( ctrl.colExpo )
		{
//			Print(sharedNorms.colExpo, "colExpo swap");
//			cout << "rightTerm = " << rightTerm << endl;
//			cout << "rightTermPrescale = " << F(rightTerm*Pow(F(2), F(sharedNorms.colExpo(k,0)))) << endl;
			Int expo = (sharedNorms.colExpo(k,0) - sharedNorms.colExpo(k-1,0));
			rightTerm *= Pow(F(2), F(expo));
//			auto bcol1 = B(ALL, IR(k-1));
//			cout << "||b_k-1|| = " << F(El::FrobeniusNorm(bcol1)) << endl;
//			auto bcol2 = B(ALL, IR(k));
//			cout << "||b_k|| = " << F(El::FrobeniusNorm(bcol2)) << endl;
//			cout << "leftTerm  = " << F(leftTerm*Pow(F(2), F(sharedNorms.colExpo(k-1,0)))) << endl;
//			cout << "rightTerm = " << F(rightTerm*Pow(F(2), F(sharedNorms.colExpo(k-1,0)))) << endl;
//			cout << endl;
//			cout << endl;
			//cout << "rho_kk    = " << rho_k_k << endl;
		}
        // NOTE: It is possible that, if delta < 1/2, that rho_k_k could be
        //       zero and the usual Lovasz condition would be satisifed.
        //       For this reason, we explicitly force a pivot if R(k,k) is
        //       deemed to be numerically zero.
		Real scale = 1;
		if( ctrl.colExpo )
			scale = Pow(F(2), F(sharedNorms.colExpo(k,0)));
        if( leftTerm <= rightTerm && rho_k_k > ctrl.zeroTol/scale )
        {
            ++k;
        }
        else
        {
            ++numSwaps;
            firstSwap = Min(firstSwap,k-1);
            if( ctrl.progress )
            {
                if( !ctrl.colExpo && rho_k_k <= ctrl.zeroTol/scale )
                    Output("Dropping from k=",k," because R(k,k) ~= 0");
                else
                    Output
                    ("Dropping from k=",k," to ",Max(k-1,1),
                     " since sqrt(delta)*R(k-1,k-1)=",leftTerm," > ",rightTerm);
            }

            ColSwap( B, k-1, k );
            if( formU )
                ColSwap( U, k-1, k );
			RowSwap( sharedNorms.colNorms, k-1, k );
			if( ctrl.colExpo )
				RowSwap( sharedNorms.colExpo, k-1, k );

            if( k == 1 )
            {
                while( true )
                {
                    // We must reinitialize since we keep k=1
                    lll::ExpandQR( 0, B, QR, t, d, sharedNorms, ctrl.numOrthog, ctrl.colExpo, ctrl.time );
                    lll::HouseholderStep( 0, QR, t, d, ctrl.time );
                    if( RealPart(QR(0,0)) <= ctrl.zeroTol/scale )
                    {
                        auto b0 = B(ALL,IR(0));
                        auto QR0 = QR(ALL,IR(0));
                        Zero( b0 );
                        Zero( QR0 );
                        t(0) = Real(2);
                        d(0) = Real(1);

                        ColSwap( B, 0, (n-1)-nullity );
                        if( formU )
                            ColSwap( U, 0, (n-1)-nullity );
						RowSwap( sharedNorms.colNorms, 0, (n-1)-nullity );
						if( ctrl.colExpo )
							RowSwap( sharedNorms.colExpo, 0, (n-1)-nullity );
                            
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
        Output("    Norm time:              ",normTimer.Total());
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

    // Keep this struct around for norm computation purposes
    // Avoid repeatedly reallocating memory
	SharedNorms<Z,F> sharedNorms;
    Zeros( sharedNorms.bcol, m, 1);
    Zeros( sharedNorms.colNorms, n, 1 );

    for (Int i=0; i<n; i++)
    {
        auto col = B( ALL, IR(i) );
        Copy(col, sharedNorms.bcol);
        sharedNorms.colNorms(i, 0) = El::FrobeniusNorm(sharedNorms.bcol);
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
            lll::ExpandQR( 0, B, QR, t, d, sharedNorms, ctrl.numOrthog, ctrl.colExpo, ctrl.time );
            lll::HouseholderStep( 0, QR, t, d, ctrl.time );
            if( RealPart(QR(0,0)) <= ctrl.zeroTol )
            {
                auto b0 = B(ALL,IR(0));
                auto QR0 = QR(ALL,IR(0));
                Zero( b0 );
                Zero( QR0 ); 
                t(0) = Real(2);
                d(0) = Real(1);

                ColSwap( B, 0, (n-1)-nullity );
                if( formU )
                    ColSwap( U, 0, (n-1)-nullity );
				RowSwap( sharedNorms.colNorms, 0, (n-1)-nullity );
                    
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
        bool zeroVector = lll::Step( k, B, U, QR, t, d, formU, sharedNorms, ctrl );
        if( zeroVector )
        {
            ColSwap( B, k, (n-1)-nullity );
            if( formU )
                ColSwap( U, k, (n-1)-nullity );
			RowSwap( sharedNorms.colNorms, k, (n-1)-nullity );
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
        Real origNorm = blas::Nrm2( rColHeight, &QR(0,k), 1 );
        Real partialNorm = origNorm;
        for( Int i=0; i<Min(k,minDim); ++i )
        {
            const Real rho_i_i = RealPart(QR(i,i));
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
				RowSwap( sharedNorms.colNorms, i, k );
                
                if( i == 0 )
                {
                    while( true )
                    {
                        // We must reinitialize since we keep k=1
                        lll::ExpandQR
                        ( 0, B, QR, t, d, sharedNorms, ctrl.numOrthog, ctrl.colExpo, ctrl.time );
                        lll::HouseholderStep( 0, QR, t, d, ctrl.time );
                        if( RealPart(QR(0,0)) <= ctrl.zeroTol )
                        {
                            auto b0 = B(ALL,IR(0));
                            auto QR0 = QR(ALL,IR(0));
                            Zero( b0 );
                            Zero( QR0 );
                            t(0) = Real(2);
                            d(0) = Real(1);

                            ColSwap( B, 0, (n-1)-nullity );
                            if( formU )
                                ColSwap( U, 0, (n-1)-nullity );
							RowSwap( sharedNorms.colNorms, 0, (n-1)-nullity );

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
                Real gamma = Abs(QR(i,k)) / partialNorm;
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

    // Keep this struct around for norm computation purposes
    // Avoid repeatedly reallocating memory
	SharedNorms<Z,F> sharedNorms;
    Zeros( sharedNorms.bcol, m, 1);
    Zeros( sharedNorms.colNorms, n, 1 );

    for (Int i=0; i<n; i++)
    {
        auto col = B( ALL, IR(i) );
        Copy(col, sharedNorms.bcol);
        sharedNorms.colNorms(i, 0) = El::FrobeniusNorm(sharedNorms.bcol);
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
            lll::ExpandQR( 0, B, QR, t, d, sharedNorms, ctrl.numOrthog, ctrl.colExpo, ctrl.time );
            lll::HouseholderStep( 0, QR, t, d, ctrl.time );
            if( RealPart(QR(0,0)) <= ctrl.zeroTol )
            {
                auto b0 = B(ALL,IR(0));
                auto QR0 = QR(ALL,IR(0));
                Zero( b0 );
                Zero( QR0 );
                t(0) = Real(2);
                d(0) = Real(1);

                ColSwap( B, 0, (n-1)-nullity );
                if( formU )
                    ColSwap( U, 0, (n-1)-nullity );
				RowSwap( sharedNorms.colNorms, 0, (n-1)-nullity );

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
        bool zeroVector = lll::Step( k, B, U, QR, t, d, formU, sharedNorms, ctrl );
        if( zeroVector )
        {
            ColSwap( B, k, (n-1)-nullity );
            if( formU )
                ColSwap( U, k, (n-1)-nullity );
			RowSwap( sharedNorms.colNorms, k, (n-1)-nullity );
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
                F dot = blas::Dot(l-i+1,&QR(i,k),1,&QR(i,l),1);
                Real nrm = blas::Nrm2(l-i+1,&QR(i,l),1);
                F mu = (dot/nrm)/nrm;
                if( ctrl.delta*Abs(RealPart(mu)) >= Real(1)/Real(2) ||
                    ctrl.delta*Abs(ImagPart(mu)) >= Real(1)/Real(2) )
                {
                    F chi = Round(mu);
                    x(l-i) = chi;
                    blas::Axpy
                    ( l+1, -chi,
                      &QR(0,l), 1,
                      &QR(0,k), 1 );
                    deepReduced = true;
                }
            }

            const Real rho_i_i = RealPart(QR(i,i));
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
                    F chi = x(l-i);
                    if( Abs(RealPart(chi)) > 0 || Abs(ImagPart(chi)) > 0 )
                    {
                        blas::Axpy
                        ( m, Z(-chi),
                          &B(0,l), 1,
                          &B(0,k), 1 );
                        if( formU )
                            blas::Axpy
                            ( n, Z(-chi),
                              &U(0,l), 1,
                              &U(0,k), 1 );
                    }
                }

                DeepColSwap( B, i, k );
                if( formU )
                    DeepColSwap( U, i, k );

                // Update the column norms
                sharedNorms.colNorms(k, 0) = sharedNorms.colNorms.Get(i, 0);
                if( ctrl.time )
                    normTimer.Start();
                sharedNorms.colNorms(i, 0) = lll::Norm2(B, sharedNorms, i, ctrl.colExpo, ctrl.time);
                if( ctrl.time )
                    normTimer.Stop();                
                
                if( i == 0 )
                {
                    while( true )
                    {
                        // We must reinitialize since we keep k=1
                        lll::ExpandQR
                        ( 0, B, QR, t, d, sharedNorms, ctrl.numOrthog, ctrl.colExpo, ctrl.time );
                        lll::HouseholderStep( 0, QR, t, d, ctrl.time );
                        if( RealPart(QR(0,0)) <= ctrl.zeroTol )
                        {
                            auto b0 = B(ALL,IR(0));
                            auto QR0 = QR(ALL,IR(0));
                            Zero( b0 );
                            Zero( QR0 );
                            t(0) = Real(2);
                            d(0) = Real(1);

                            ColSwap( B, 0, (n-1)-nullity );
                            if( formU )
                                ColSwap( U, 0, (n-1)-nullity );
							RowSwap( sharedNorms.colNorms, 0, (n-1)-nullity );
							
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
