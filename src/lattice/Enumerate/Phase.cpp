/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

namespace El {

namespace svp {

// NOTE: This accepts the N from the Q(DN), not R from QR, where R=DN
template<typename F>
void CoordinatesToSparse
( const Matrix<F>& N,
  const Matrix<F>& v,
        Matrix<F>& y )
{
    DEBUG_ONLY(CSE cse("svp::CoordinatesToSparse"))
    y = v;
    Trmv( UPPER, NORMAL, UNIT, N, y );
    Round( y );
}

template<typename F>
void TransposedCoordinatesToSparse
( const Matrix<F>& NTrans,
  const Matrix<F>& v,
        Matrix<F>& y )
{
    DEBUG_ONLY(CSE cse("svp::TransposedCoordinatesToSparse"))
    y = v;
    Trmv( LOWER, TRANSPOSE, UNIT, NTrans, y );
    Round( y );
}

template<typename F>
void BatchCoordinatesToSparse
( const Matrix<F>& N,
  const Matrix<F>& V,
        Matrix<F>& Y )
{
    DEBUG_ONLY(CSE cse("svp::BatchCoordinatesToSparse"))
    Y = V;
    Trmm( LEFT, UPPER, NORMAL, UNIT, F(1), N, Y );
    Round( Y );
}

template<typename F>
void BatchTransposedCoordinatesToSparse
( const Matrix<F>& NTrans,
  const Matrix<F>& V,
        Matrix<F>& Y )
{
    DEBUG_ONLY(CSE cse("svp::BatchTransposedCoordinatesToSparse"))
    Y = V;
    Trmm( LEFT, LOWER, TRANSPOSE, UNIT, F(1), NTrans, Y );
    Round( Y );
}

template<typename F>
void SparseToCoordinates
( const Matrix<F>& N,
  const Matrix<F>& y,
        Matrix<F>& v )
{
    DEBUG_ONLY(CSE cse("svp::SparseToCoordinates"))
    const Int n = N.Height();

    v = y;

    // A custom rounded analogue of an upper-triangular solve
    const F* NBuf = N.LockedBuffer();
          F* vBuf = v.Buffer();
    const Int NLDim = N.LDim();
    for( Int j=n-1; j>=0; --j )
    {
        F tau = 0;
        for( Int k=j+1; k<n; ++k ) 
            tau += NBuf[j+k*NLDim]*vBuf[k];
        vBuf[j] -= Round(tau);
    }
}

template<typename F>
void TransposedSparseToCoordinates
( const Matrix<F>& NTrans,
  const Matrix<F>& y,
        Matrix<F>& v )
{
    DEBUG_ONLY(CSE cse("svp::TransposedSparseToCoordinates"))
    const Int n = NTrans.Height();

    v = y;

    // A custom rounded analogue of an upper-triangular solve
    const F* NTransBuf = NTrans.LockedBuffer();
          F* vBuf = v.Buffer();
    const Int NTransLDim = NTrans.LDim();
    for( Int j=n-1; j>=0; --j )
    {
        const F* nBuf = &NTransBuf[j*NTransLDim];

        F tau = 0;
        for( Int k=j+1; k<n; ++k ) 
            tau += nBuf[k]*vBuf[k];
        vBuf[j] -= Round(tau);
    }
}

// TODO: Optimize this routine by changing the loop order?
template<typename F>
void BatchSparseToCoordinatesUnblocked
( const Matrix<F>& N,
  const Matrix<F>& Y,
        Matrix<F>& V )
{
    DEBUG_ONLY(CSE cse("svp::BatchSparseToCoordinatesUnblocked"))
    const Int n = N.Height();
    const Int numRHS = Y.Width();

    // A custom rounded analogue of an upper-triangular solve
    const F* NBuf = N.LockedBuffer();
    const F* YBuf = Y.LockedBuffer();
          F* VBuf = V.Buffer();
    const Int NLDim = N.LDim();
    const Int YLDim = Y.LDim();
    const Int VLDim = V.LDim();
    for( Int l=0; l<numRHS; ++l )
    {
              F* vBuf = &VBuf[l*VLDim];
        const F* yBuf = &YBuf[l*YLDim];
        for( Int j=n-1; j>=0; --j )
        {
            F tau = 0;
            for( Int k=j+1; k<n; ++k ) 
                tau += NBuf[j+k*NLDim]*vBuf[k];
            vBuf[j] = yBuf[j] + Round(vBuf[j]-tau); 
        }
    }
}

template<typename F>
void BatchTransposedSparseToCoordinatesUnblocked
( const Matrix<F>& NTrans,
  const Matrix<F>& Y,
        Matrix<F>& V )
{
    DEBUG_ONLY(CSE cse("svp::BatchTransposedSparseToCoordinatesUnblocked"))
    const Int n = NTrans.Height();
    const Int numRHS = Y.Width();

    // A custom rounded analogue of an upper-triangular solve
    const F* NTransBuf = NTrans.LockedBuffer();
    const F* YBuf = Y.LockedBuffer();
          F* VBuf = V.Buffer();
    const Int NTransLDim = NTrans.LDim();
    const Int YLDim = Y.LDim();
    const Int VLDim = V.LDim();
    for( Int l=0; l<numRHS; ++l )
    {
              F* vBuf = &VBuf[l*VLDim];
        const F* yBuf = &YBuf[l*YLDim];
        for( Int j=n-1; j>=0; --j )
        {
            const F* nBuf = &NTransBuf[j*NTransLDim];

            F tau = 0;
            for( Int k=j+1; k<n; ++k ) 
                tau += nBuf[k]*vBuf[k];
            vBuf[j] = yBuf[j] + Round(vBuf[j]-tau); 
        }
    }
}

// TODO: Optimize this routine
template<typename F>
void BatchSparseToCoordinates
( const Matrix<F>& N,
  const Matrix<F>& Y,
        Matrix<F>& V,
        Int blocksize )
{
    DEBUG_ONLY(CSE cse("svp::BatchSparseToCoordinates"))
    const Int n = N.Height();
    const Int numRHS = Y.Width();

    // TODO: Test the relative performance of this branch
    if( numRHS == 1 )
    {
        SparseToCoordinates( N, Y, V );
        return;
    }

    Zeros( V, n, numRHS );
    const Int iLast = LastOffset( n, blocksize );
    for( Int i=iLast; i>=0; i-=blocksize )
    { 
        const Int nb = Min(n-i,blocksize);
        const Range<Int> ind0(0,i), ind1(i,i+nb);
        
        auto N01 = N( ind0, ind1 );
        auto N11 = N( ind1, ind1 );
        auto Y1 = Y( ind1, ALL );
        auto V0 = V( ind0, ALL );
        auto V1 = V( ind1, ALL );

        BatchSparseToCoordinatesUnblocked( N11, Y1, V1 );
        Gemm( NORMAL, NORMAL, F(-1), N01, V1, F(1), V0 );
    }
}

template<typename F>
void BatchTransposedSparseToCoordinates
( const Matrix<F>& NTrans,
  const Matrix<F>& Y,
        Matrix<F>& V,
        Int blocksize )
{
    DEBUG_ONLY(CSE cse("svp::BatchTransposedSparseToCoordinates"))
    const Int n = NTrans.Height();
    const Int numRHS = Y.Width();

    // TODO: Test the relative performance of this branch
    if( numRHS == 1 )
    {
        TransposedSparseToCoordinates( NTrans, Y, V );
        return;
    }

    Zeros( V, n, numRHS );
    const Int iLast = LastOffset( n, blocksize );
    for( Int i=iLast; i>=0; i-=blocksize )
    { 
        const Int nb = Min(n-i,blocksize);
        const Range<Int> ind0(0,i), ind1(i,i+nb);
        
        auto NTrans10 = NTrans( ind1, ind0 );
        auto NTrans11 = NTrans( ind1, ind1 );
        auto Y1 = Y( ind1, ALL );
        auto V0 = V( ind0, ALL );
        auto V1 = V( ind1, ALL );

        BatchTransposedSparseToCoordinatesUnblocked( NTrans11, Y1, V1 );
        Gemm( TRANSPOSE, NORMAL, F(-1), NTrans10, V1, F(1), V0 );
    }
}

template<typename F>
Base<F> CoordinatesToNorm
( const Matrix<Base<F>>& d,
  const Matrix<F>& N,
  const Matrix<F>& v )
{
    DEBUG_ONLY(CSE cse("svp::CoordinatesToNorm"))
    Matrix<F> z( v );
    Trmv( UPPER, NORMAL, UNIT, N, z );
    DiagonalScale( LEFT, NORMAL, d, z );
    return FrobeniusNorm( z );
}

template<typename F>
Base<F> TransposedCoordinatesToNorm
( const Matrix<Base<F>>& d,
  const Matrix<F>& NTrans,
  const Matrix<F>& v )
{
    DEBUG_ONLY(CSE cse("svp::TransposedCoordinatesToNorm"))
    Matrix<F> z( v );
    Trmv( LOWER, TRANSPOSE, UNIT, NTrans, z );
    DiagonalScale( LEFT, NORMAL, d, z );
    return FrobeniusNorm( z );
}

template<typename F>
Matrix<Base<F>> NestedColumnTwoNorms( const Matrix<F>& Z, Int numNested=1 )
{
    DEBUG_ONLY(CSE cse("svp::NestedColumnTwoNorms"))
    typedef Base<F> Real;
    const Int n = Z.Height();
    const Int numRHS = Z.Width();

    Matrix<Real> colNorms(numRHS,numNested);
    // Compute nested norms in linear time
    for( Int j=0; j<numRHS; ++j )
    {
        const F* zBuf = Z.LockedBuffer(0,j);
        Real scale=0, scaledSquare=1;          
        for( Int i=n-1; i>=numNested; --i )
            UpdateScaledSquare( zBuf[i], scale, scaledSquare );
        for( Int i=numNested-1; i>=0; --i )
        {
            UpdateScaledSquare( zBuf[i], scale, scaledSquare );
            colNorms.Set( j, i, scale*Sqrt(scaledSquare) );
        }
    }
    return colNorms;
}

template<typename F>
Matrix<Base<F>> BatchCoordinatesToNorms
( const Matrix<Base<F>>& d,
  const Matrix<F>& N,
  const Matrix<F>& V,
        Int numNested=1 )
{
    DEBUG_ONLY(CSE cse("svp::BatchCoordinatesToNorms"))
    Matrix<F> Z( V );
    // TODO: Decide whether this branch is necessary or not...
    if( V.Width() == 1 )
        Trmv( UPPER, NORMAL, UNIT, N, Z );
    else
        Trmm( LEFT, UPPER, NORMAL, UNIT, F(1), N, Z );
    DiagonalScale( LEFT, NORMAL, d, Z );

    return NestedColumnTwoNorms( Z, numNested );
}

template<typename F>
Matrix<Base<F>> BatchTransposedCoordinatesToNorms
( const Matrix<Base<F>>& d,
  const Matrix<F>& NTrans,
  const Matrix<F>& V,
        Int numNested=1 )
{
    DEBUG_ONLY(CSE cse("svp::BatchTransposedCoordinatesToNorms"))
    Matrix<F> Z( V );
    // TODO: Decide whether this branch is necessary or not...
    if( V.Width() == 1 )
        Trmv( LOWER, TRANSPOSE, UNIT, NTrans, Z );
    else
        Trmm( LEFT, LOWER, TRANSPOSE, UNIT, F(1), NTrans, Z );
    DiagonalScale( LEFT, NORMAL, d, Z );

    return NestedColumnTwoNorms( Z, numNested );
}

template<typename F>
Base<F> SparseToNormLowerBound
( const Matrix<Base<F>>& d,
  const Matrix<F>& y )
{
    DEBUG_ONLY(CSE cse("svp::SparseToNormLowerBound"))
    typedef Base<F> Real;
    const Int n = d.Height();
    const Real* dBuf = d.LockedBuffer();
    const F* yBuf = y.LockedBuffer();

    const Real oneHalf = Real(1)/Real(2);

    Real lowerBoundSquared = 0;
    for( Int j=0; j<n; ++j )
    {
        if( yBuf[j] != F(0) )
        {
            const Real arg = (Abs(yBuf[j])-oneHalf)*dBuf[j];
            lowerBoundSquared += Pow(arg,Real(2));
        }
    }
    return Sqrt(lowerBoundSquared);
}

template<typename F>
Matrix<Base<F>> BatchSparseToNormLowerBound
( const Matrix<Base<F>>& d,
  const Matrix<F>& Y )
{
    DEBUG_ONLY(CSE cse("svp::BatchSparseToNormLowerBound"))
    typedef Base<F> Real;
    const Int numRHS = Y.Width();
    Matrix<Real> normBounds;
    Zeros( normBounds, numRHS, 1 );
    for( Int j=0; j<numRHS; ++j )
        normBounds.Set( j, 0, SparseToNormLowerBound(d,Y(ALL,IR(j))) );
    return normBounds;
}

template<typename F>
Base<F> SparseToNorm
( const Matrix<Base<F>>& d,
  const Matrix<F>& N,
  const Matrix<F>& y )
{
    DEBUG_ONLY(CSE cse("svp::SparseToNorm"))
    Matrix<F> v;
    SparseToCoordinates( N, y, v );
    return CoordinatesToNorm( d, N, v );
}

template<typename F>
Base<F> TransposedSparseToNorm
( const Matrix<Base<F>>& d,
  const Matrix<F>& NTrans,
  const Matrix<F>& y )
{
    DEBUG_ONLY(CSE cse("svp::TransposedSparseToNorm"))
    Matrix<F> v;
    TransposedSparseToCoordinates( NTrans, y, v );
    return TransposedCoordinatesToNorm( d, NTrans, v );
}

template<typename F>
Matrix<Base<F>> BatchSparseToNorm
( const Matrix<Base<F>>& d,
  const Matrix<F>& N,
  const Matrix<F>& Y,
        Int blocksize,
        Int numNested=1 )
{
    DEBUG_ONLY(CSE cse("svp::BatchSparseToNorm"))
    Matrix<F> V;
    BatchSparseToCoordinates( N, Y, V, blocksize );
    return BatchCoordinatesToNorms( d, N, V, numNested );
}

template<typename F>
Matrix<Base<F>> BatchTransposedSparseToNorm
( const Matrix<Base<F>>& d,
  const Matrix<F>& NTrans,
  const Matrix<F>& Y,
        Int blocksize,
        Int numNested=1 )
{
    DEBUG_ONLY(CSE cse("svp::BatchTransposedSparseToNorm"))
    Matrix<F> V;
    BatchTransposedSparseToCoordinates( NTrans, Y, V, blocksize );
    return BatchTransposedCoordinatesToNorms( d, NTrans, V, numNested );
}

template<typename Real>
class PhaseEnumerationCache
{
private:
    const Matrix<Real>& B_;
    const Matrix<Real>& d_;
    const Matrix<Real>& N_;
          Matrix<Real> NTrans_;
          Matrix<Real> normUpperBounds_;
    bool foundVector_=false;

    Int numQueued_=0;
    Matrix<Real> Y_;
    Matrix<Real> VCand_;

    Int insertionBound_;
    Matrix<Real> v_;

    Int blocksize_=32;
    bool useTranspose_=true;

public:
    PhaseEnumerationCache
    ( const Matrix<Real>& B,
      const Matrix<Real>& d,
      const Matrix<Real>& N,
      const Matrix<Real>& normUpperBounds,
            Int batchSize=256,
            Int blocksize=32,
            bool useTranspose=true )
    : B_(B),
      d_(d),
      N_(N),
      normUpperBounds_(normUpperBounds),
      foundVector_(false),
      numQueued_(0),
      insertionBound_(normUpperBounds.Height()),
      blocksize_(blocksize),
      useTranspose_(useTranspose)
    { 
        Zeros( Y_, N.Height(), batchSize );   
        if( useTranspose )
            Transpose( N, NTrans_ );
    }

    Int Height() const { return N_.Height(); }

    bool FoundVector() const { return foundVector_; }
    const Matrix<Real>& BestVector() const { return v_; }

    Int InsertionIndex() const
    {
        if( foundVector_ )
        {
            return insertionBound_-1;
        }
        else
        {
            LogicError("Did not find a shorter vector");
            return -1;
        }
    }

    Real InsertionNorm() const
    {
        const Int insertionIndex = InsertionIndex();
        return normUpperBounds_.Get(insertionIndex,0);
    }

    void Flush()
    {
        if( numQueued_ == 0 )
            return;

        auto YActive = Y_( ALL, IR(0,numQueued_) );

        Matrix<Real> colNorms;
        if( useTranspose_ )
        {
            // TODO: Add this as an option
            /*
            Timer timer;
            timer.Start();
            BatchTransposedSparseToCoordinates
            ( NTrans_, YActive, VCand_, blocksize_ );
            const double transformTime = timer.Stop();
            const double n = YActive.Height();
            const double transformGflops = double(numQueued_)*n*n/(1.e9*transformTime);
            Output(numQueued_," transforms: ",timer.Stop()," seconds (",transformGflops," GFlop/s");
            timer.Start();
            colNorms =
              BatchTransposedCoordinatesToNorms
              ( d_, NTrans_, VCand_, insertionBound_ );
            const double normTime = timer.Stop();
            const double normGflops = double(numQueued_)*n*n/(1.e9*normTime);
            Output(numQueued_," norms: ",timer.Stop()," seconds (",normGflops," GFlop/s");
            */

            BatchTransposedSparseToCoordinates
            ( NTrans_, YActive, VCand_, blocksize_ );
            colNorms =
              BatchTransposedCoordinatesToNorms
              ( d_, NTrans_, VCand_, insertionBound_ );
        }
        else
        {
            BatchSparseToCoordinates( N_, YActive, VCand_, blocksize_ );
            colNorms =
              BatchCoordinatesToNorms( d_, N_, VCand_, insertionBound_ );
        }

        for( Int j=0; j<numQueued_; ++j )
        {
            for( Int k=0; k<insertionBound_; ++k )
            {
                const Real bNorm = colNorms.Get(j,k);
                if( bNorm < normUpperBounds_.Get(k,0) && bNorm != Real(0) )
                {
                    const Range<Int> subInd(k,END);

                    auto y = YActive(subInd,IR(j));
                    auto vCand = VCand_(subInd,IR(j));

                    Output
                    ("normUpperBound=",normUpperBounds_.Get(k,0),
                     ", bNorm=",bNorm,", k=",k);
                    Print( y, "y" );

                    // Check that the reverse transformation holds
                    Matrix<Real> yCheck;
                    CoordinatesToSparse( N_(subInd,subInd), vCand, yCheck );
                    yCheck -= y;
                    if( FrobeniusNorm(yCheck) != Real(0) )
                    {
                        Print( B_(ALL,subInd), "B" );
                        Print( d_(subInd,ALL), "d" );
                        Print( N_(subInd,subInd), "N" );
                        Print( vCand, "vCand" );
                        Print( yCheck, "eCheck" );
                        LogicError("Invalid sparse transformation");
                    }

                    Copy( vCand, v_ );
                    Print( v_, "v" );

                    Matrix<Real> b;
                    Zeros( b, B_.Height(), 1 );
                    Gemv( NORMAL, Real(1), B_(ALL,subInd), v_, Real(0), b );
                    Print( b, "b" );

                    normUpperBounds_.Set(k,0,bNorm);
                    foundVector_ = true;
                    insertionBound_ = k+1;
                }
                // TODO: Keep track of 'stock' vectors?
            }
        }
        numQueued_ = 0;
        Zero( Y_ );
    }

    void Enqueue( const Matrix<Real>& y )
    {
        MemCopy( Y_.Buffer(0,numQueued_), y.LockedBuffer(), y.Height() ); 

        ++numQueued_;
        if( numQueued_ == Y_.Width() )
            Flush();
    }

    void Enqueue( const vector<pair<Int,Int>>& y )
    {
        Real* yBuf = Y_.Buffer(0,numQueued_);

        const Int numEntries = y.size();
        for( Int e=0; e<numEntries; ++e )
            yBuf[y[e].first] = Real(y[e].second);

        ++numQueued_;
        if( numQueued_ == Y_.Width() )
            Flush();
    }

    void Enqueue( const vector<pair<Int,Real>>& y )
    {
        Real* yBuf = Y_.Buffer(0,numQueued_);

        const Int numEntries = y.size();
        for( Int e=0; e<numEntries; ++e )
            yBuf[y[e].first] = y[e].second;

        ++numQueued_;
        if( numQueued_ == Y_.Width() )
            Flush();
    }

    void MaybeEnqueue( const vector<pair<Int,Int>>& y, double enqueueProb=1. )
    {
        if( enqueueProb >= 1. || SampleUniform<double>(0,1) <= enqueueProb )
            Enqueue( y );
    }

    ~PhaseEnumerationCache() { }
};

template<typename Real>
struct PhaseEnumerationCtrl
{
  const vector<Int>& phaseOffsets;
  const vector<Int>& minInfNorms;
  const vector<Int>& maxInfNorms;
  const vector<Int>& minOneNorms;
  const vector<Int>& maxOneNorms;
  const double enqueueProb=1.;
  const bool earlyExit=false;
  const Int progressLevel=4;

  PhaseEnumerationCtrl
  ( const vector<Int>& offsets,
    const vector<Int>& minInf,
    const vector<Int>& maxInf,
    const vector<Int>& minOne,
    const vector<Int>& maxOne,
    double accept=1.,
    bool early=false,
    Int level=4 )
  : phaseOffsets(offsets),
    minInfNorms(minInf), maxInfNorms(maxInf),
    minOneNorms(minOne), maxOneNorms(maxOne),
    enqueueProb(accept),
    earlyExit(early),
    progressLevel(level)
  { }
};

// TODO: Reverse the enumeration order since nonzeros are more likely to happen
//       at the bottom of the vector. This will require changing the arguments
//       to this routine.
template<typename Real>
void PhaseEnumerationLeafInner
(       PhaseEnumerationCache<Real>& cache,
  const PhaseEnumerationCtrl<Real>& ctrl,
        vector<pair<Int,Int>>& y,
  const Int beg,
  const Int baseInfNorm,
  const Int baseOneNorm,
  const bool zeroSoFar )
{
    DEBUG_ONLY(CSE cse("svp::PhaseEnumerationLeafInner"))
    const Int n = ctrl.phaseOffsets.back();
    const Int minInf = ctrl.minInfNorms.back();
    const Int maxInf = ctrl.maxInfNorms.back();
    const Int minOne = ctrl.minOneNorms.back();
    const Int maxOne = ctrl.maxOneNorms.back();

    const Int bound = Min(maxInf,maxOne-baseOneNorm);
    for( Int absBeta=1; absBeta<=bound; ++absBeta )
    {
        const Int newOneNorm = baseOneNorm + absBeta;
        const Int newInfNorm = Max(baseInfNorm,absBeta);

        if( newOneNorm >= minOne && newInfNorm >= minInf )
        {
            // We can insert +-|beta| into any position and be admissible,
            // so enqueue each possibility
            for( Int i=beg; i<n; ++i )
            {
                y.emplace_back( i, absBeta );
                cache.MaybeEnqueue( y, ctrl.enqueueProb );

                if( !zeroSoFar )
                {
                    y.back() = pair<Int,Int>(i,-absBeta);
                    cache.MaybeEnqueue( y, ctrl.enqueueProb );
                }

                y.pop_back();
            }
        }

        if( newOneNorm < maxOne )
        {
            // We can insert +-|beta| into any position and still have room
            // left for the one norm bound, so do so and then recurse
            for( Int i=beg; i<n-1; ++i )
            {
                y.emplace_back( i, absBeta );
                PhaseEnumerationLeafInner
                ( cache, ctrl, y, i+1, newInfNorm, newOneNorm, false );

                if( !zeroSoFar )
                {
                    y.back().second = -absBeta;
                    PhaseEnumerationLeafInner
                    ( cache, ctrl, y, i+1, newInfNorm, newOneNorm, false );
                }

                y.pop_back();
            }
        }
    }
}

template<typename Real>
void PhaseEnumerationLeaf
(       PhaseEnumerationCache<Real>& cache,
  const PhaseEnumerationCtrl<Real>& ctrl,
        vector<pair<Int,Int>>& y,
  const bool zeroSoFar )
{
    DEBUG_ONLY(CSE cse("svp::PhaseEnumerationLeaf"))

    // Enqueue the zero phase if it is admissible
    if( ctrl.minInfNorms.back() == Int(0) &&
        ctrl.minOneNorms.back() == Int(0) )
        cache.MaybeEnqueue( y, ctrl.enqueueProb );

    const Int beg = ctrl.phaseOffsets[ctrl.phaseOffsets.size()-2];
    const Int baseInfNorm = 0;
    const Int baseOneNorm = 0;
    PhaseEnumerationLeafInner
    ( cache, ctrl, y, beg, baseInfNorm, baseOneNorm, zeroSoFar );
}

// TODO: Reverse the enumeration order since nonzeros are more likely to happen
//       at the bottom of the vector. This will require changing the arguments
//       to this routine.
template<typename Real>
void PhaseEnumerationNodeInner
(       PhaseEnumerationCache<Real>& cache,
  const PhaseEnumerationCtrl<Real>& ctrl,
        vector<pair<Int,Int>>& y,
  const Int phase,
  const Int beg,
  const Int baseInfNorm,
  const Int baseOneNorm,
  const bool zeroSoFar )
{
    DEBUG_ONLY(CSE cse("svp::PhaseEnumerationNodeInner"))
    const Int n = cache.Height();
    if( ctrl.earlyExit && cache.FoundVector() )
        return;

    const Int phaseBeg = ctrl.phaseOffsets[phase];
    const Int nextPhaseBeg = ctrl.phaseOffsets[phase+1];
    const Int minInf = ctrl.minInfNorms[phase];
    const Int maxInf = ctrl.maxInfNorms[phase];
    const Int minOne = ctrl.minOneNorms[phase];
    const Int maxOne = ctrl.maxOneNorms[phase];

    if( nextPhaseBeg >= n )
    {
        PhaseEnumerationLeaf( cache, ctrl, y, zeroSoFar );
        return;
    }

    if( beg == phaseBeg && minInf == 0 && minOne == 0 )
    {
        // This phase can be all zeroes, so move to the next phase
        PhaseEnumerationNode( cache, ctrl, y, phase+1, zeroSoFar );
    }

    const Int bound = Min(maxInf,maxOne-baseOneNorm);
    for( Int absBeta=1; absBeta<=bound; ++absBeta ) 
    {
        const Int phaseOneNorm = baseOneNorm + absBeta;
        const Int phaseInfNorm = Max( baseInfNorm, absBeta );

        if( phaseOneNorm >= minOne && phaseInfNorm >= minInf )
        {
            // We could insert +-|beta| into any position and still have
            // an admissible choice for the current phase, so procede to
            // the next phase with each such choice

            for( Int i=beg; i<nextPhaseBeg; ++i )
            {
                // Fix y[i] = +|beta| and move to the next phase
                y.emplace_back( i, absBeta );
                if( phase < ctrl.progressLevel )
                    Output("phase ",phase,": y[",i,"]=",absBeta);
                PhaseEnumerationNode( cache, ctrl, y, phase+1, false );

                if( !zeroSoFar )
                {
                    // Fix y[i] = -|beta| and move to the next phase
                    y.back().second = -absBeta;
                    if( phase < ctrl.progressLevel )
                        Output("phase ",phase,": y[",i,"]=",-absBeta);
                    PhaseEnumerationNode( cache, ctrl, y, phase+1, false );
                }
                
                y.pop_back();
            }
        }

        if( phaseOneNorm < maxOne )
        {
            // Inserting +-|beta| into any position still leaves us with
            // room in the current phase

            for( Int i=beg; i<nextPhaseBeg; ++i )
            {
                // Fix y[i] = +|beta| and move to y[i+1]
                y.emplace_back( i, absBeta ); 
                PhaseEnumerationNodeInner
                ( cache, ctrl, y, 
                  phase, i+1,
                  phaseInfNorm, phaseOneNorm, false );

                if( !zeroSoFar )
                {
                    // Fix y[i] = -|beta| and move to y[i+1]
                    y.back().second = -absBeta;
                    PhaseEnumerationNodeInner
                    ( cache, ctrl, y,
                      phase, i+1,
                      phaseInfNorm, phaseOneNorm, false );
                }
               
                y.pop_back();
            }
        }
    }
}

template<typename Real>
void PhaseEnumerationNode
(       PhaseEnumerationCache<Real>& cache,
  const PhaseEnumerationCtrl<Real>& ctrl,
        vector<pair<Int,Int>>& y,
  const Int phase,
  const bool zeroSoFar )
{
    DEBUG_ONLY(CSE cse("svp::PhaseEnumerationNode"))
    const Int baseInfNorm = 0;
    const Int baseOneNorm = 0;
    PhaseEnumerationNodeInner
    ( cache, ctrl, y,
      phase, ctrl.phaseOffsets[phase],
      baseInfNorm, baseOneNorm, zeroSoFar ); 
}

template<typename Real>
pair<Real,Int>
PhaseEnumeration
( const Matrix<Real>& B,
  const Matrix<Real>& d,
  const Matrix<Real>& N,
  const Matrix<Real>& normUpperBounds,
        Int startIndex,
        Int phaseLength,
        double enqueueProb,
  const vector<Int>& minInfNorms,
  const vector<Int>& maxInfNorms,
  const vector<Int>& minOneNorms,
  const vector<Int>& maxOneNorms,
        Matrix<Real>& v,
        Int progressLevel )
{
    DEBUG_ONLY(CSE cse("svp::PhaseEnumeration"))
    const Int n = N.Height();
    if( n <= 1 )
        return pair<Real,Int>(2*normUpperBounds.Get(0,0)+1,0);

    // TODO: Make starting index modifiable
    const Int numPhases = ((n-startIndex)+phaseLength-1)/phaseLength;
    if( numPhases != Int(maxInfNorms.size()) )
        LogicError("Invalid length of maxInfNorms");
    if( numPhases != Int(maxOneNorms.size()) )
        LogicError("Invalid length of maxOneNorms");

    // TODO: Loop and increase bands for min and max one and inf norms?

    // NOTE: The blocking doesn't seem to help the performance (yet)
    const Int batchSize = 512;
    const Int blocksize = 32;
    const bool useTranspose = true;
    PhaseEnumerationCache<Real>
      cache( B, d, N, normUpperBounds, batchSize, blocksize, useTranspose );

    const bool earlyExit = false;

    // TODO: Allow phaseOffsets to be an input
    vector<Int> phaseOffsets(numPhases+1);
    for( Int phase=0; phase<=numPhases; ++phase )
        phaseOffsets[phase] = Min(startIndex+phase*phaseLength,n);

    PhaseEnumerationCtrl<Real>
      ctrl
      (phaseOffsets,
       minInfNorms,maxInfNorms,
       minOneNorms,maxOneNorms,
       enqueueProb,
       earlyExit,progressLevel);

    vector<pair<Int,Int>> y;
    Int phase=0;
    bool zeroSoFar = true;
    PhaseEnumerationNode( cache, ctrl, y, phase, zeroSoFar );

    cache.Flush();

    if( cache.FoundVector() )
    {
        v = cache.BestVector();
        const Real insertionNorm = cache.InsertionNorm();
        const Int insertionIndex = cache.InsertionIndex();
        return pair<Real,Int>(insertionNorm,insertionIndex);
    }
    else
    {
        return pair<Real,Int>(2*normUpperBounds.Get(0,0)+1,0);
    }
}

template<typename Real>
Real PhaseEnumeration
( const Matrix<Real>& B,
  const Matrix<Real>& d,
  const Matrix<Real>& N,
        Real normUpperBound,
        Int startIndex,
        Int phaseLength,
        double enqueueProb,
  const vector<Int>& minInfNorms,
  const vector<Int>& maxInfNorms,
  const vector<Int>& minOneNorms,
  const vector<Int>& maxOneNorms,
        Matrix<Real>& v,
        Int progressLevel )
{
    DEBUG_ONLY(CSE cse("svp::PhaseEnumeration"))
    Matrix<Real> normUpperBounds(1,1);
    normUpperBounds.Set(0,0,normUpperBound);
    auto pair = 
      PhaseEnumeration
      ( B, d, N, normUpperBounds,
        startIndex, phaseLength, enqueueProb,
        minInfNorms, maxInfNorms,
        minOneNorms, maxOneNorms,
        v, progressLevel );
    return pair.first;
}

} // namespace svp

// TODO: Instantiate batched variants?
#define PROTO(F) \
  template void svp::CoordinatesToSparse \
  ( const Matrix<F>& N, \
    const Matrix<F>& v, \
          Matrix<F>& y ); \
  template void svp::TransposedCoordinatesToSparse \
  ( const Matrix<F>& NTrans, \
    const Matrix<F>& v, \
          Matrix<F>& y ); \
  template void svp::SparseToCoordinates \
  ( const Matrix<F>& N, \
    const Matrix<F>& y, \
          Matrix<F>& v ); \
  template void svp::TransposedSparseToCoordinates \
  ( const Matrix<F>& NTrans, \
    const Matrix<F>& y, \
          Matrix<F>& v ); \
  template Base<F> svp::CoordinatesToNorm \
  ( const Matrix<Base<F>>& d, \
    const Matrix<F>& N, \
    const Matrix<F>& v ); \
  template Base<F> svp::TransposedCoordinatesToNorm \
  ( const Matrix<Base<F>>& d, \
    const Matrix<F>& NTrans, \
    const Matrix<F>& v ); \
  template Base<F> svp::SparseToNorm \
  ( const Matrix<Base<F>>& d, \
    const Matrix<F>& N, \
    const Matrix<F>& y ); \
  template Base<F> svp::TransposedSparseToNorm \
  ( const Matrix<Base<F>>& d, \
    const Matrix<F>& NTrans, \
    const Matrix<F>& y ); \
  template Base<F> svp::PhaseEnumeration \
  ( const Matrix<F>& B, \
    const Matrix<Base<F>>& d, \
    const Matrix<F>& N, \
          Base<F> normUpperBound, \
          Int startIndex, \
          Int phaseLength, \
          double enqueueProb, \
    const vector<Int>& minInfNorms, \
    const vector<Int>& maxInfNorms, \
    const vector<Int>& minOneNorms, \
    const vector<Int>& maxOneNorms, \
          Matrix<F>& v, \
          Int progressLevel ); \
  template pair<Base<F>,Int> svp::PhaseEnumeration \
  ( const Matrix<F>& B, \
    const Matrix<Base<F>>& d, \
    const Matrix<F>& N, \
    const Matrix<Base<F>>& normUpperBounds, \
          Int startIndex, \
          Int phaseLength, \
          double enqueueProb, \
    const vector<Int>& minInfNorms, \
    const vector<Int>& maxInfNorms, \
    const vector<Int>& minOneNorms, \
    const vector<Int>& maxOneNorms, \
          Matrix<F>& v, \
          Int progressLevel );

#define EL_NO_INT_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#define EL_NO_COMPLEX_PROTO
#include <El/macros/Instantiate.h>

} // namespace El
