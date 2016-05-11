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

// See Algorithm 2 from:
//
//   Nicolas Gama, Phong Q. Nguyen, and Oded Regev,
//   "Lattice enumeration using extreme pruning", Eurocrypt 2010.
//
// Note that our algorithm uses 'R' to denote the upper triangular matrix from
// the QR factorization of B and 'u' to denote the sequence of upper-bounds
//
//     u(0)^2 <= u(1)^2 <= ... <= u(n-1)^2.
//
//
// TODO: Extend the following to handle complex lattices.

namespace gnr_enum {

template<typename Real,typename=EnableIf<IsReal<Real>>>
Real Helper
( const Matrix<Real>& d,
  const Matrix<Real>& N,
  const Matrix<Real>& u,
        Matrix<Real>& v,
  const EnumCtrl<Real>& ctrl )
{
    DEBUG_ONLY(CSE cse("svp::gnr_enum::Helper"))
    const Int n = N.Height();
    if( n != N.Width() )
        LogicError("Expected height(N) = width(N)");

    Matrix<Real> S;
    Zeros( S, n+1, n );

    // The 'indices' vector is indexed differently from the 'r' from
    // Gama/Nguyen/Regev, and (r_0,r_1,r_2,...,r_n)=(0,1,2,...,n) becomes
    // (-1,0,1,...,n-1).
    Matrix<Int> indices;
    Zeros( indices, n+1, 1 );
    for( Int j=0; j<=n; ++j )
        indices(j) = j-1;

    // Note: We maintain the norms rather than their squares
    Matrix<Real> partialNorms;
    Zeros( partialNorms, n+1, 1 );

    Zeros( v, n, 1 );
    if( n == 0 )
        return Real(0);
    v(0) = Real(1);
    Real* vBuf = &v(0);

    Matrix<Real> centers;
    Zeros( centers, n, 1 );

    Matrix<Real> jumps;
    Zeros( jumps, n, 1 );

    Int lastNonzero = 0; // -1 if all indices are zero

    Int k=0;
    while( true )
    {
        Real diff = vBuf[k]-centers(k);
        Real rho_k = lapack::SafeNorm( partialNorms(k+1), diff*d(k) );
        partialNorms(k) = rho_k;
        if( rho_k < u((n-1)-k) )
        {
            if( k == 0 )
            {
                // Success
                return rho_k;
            }
            else
            {
                // Move down the tree
                --k;
                indices(k) = Max(indices(k),indices(k+1));

                Real* s = &S(0,k);
                for( Int i=indices(k+1); i>=k+1; --i )
                    s[i] = s[i+1] + vBuf[i]*N(k,i);

                centers(k) = -S(k+1,k);
                vBuf[k] = Round(centers(k));
                jumps(k) = Real(1);
            }
        }
        else
        {
            // Move up the tree
            ++k;
            if( k == n )
                return 2*u(n-1)+1; // An arbitrary value > than u(n-1)
            indices(k) = k; // indicate that (i,j) are not synchronized
            if( k >= lastNonzero )
            {
                if( ctrl.innerProgress )
                    Output("lastNonzero: ",k);
                lastNonzero = k;
                vBuf[k] += Real(1);
            }
            else
            {
                if( vBuf[k] > centers(k) )
                    vBuf[k] -= jumps(k);
                else
                    vBuf[k] += jumps(k);
                jumps(k) += Real(1);
            }
        }
    }
}

template<typename Real,typename=EnableIf<IsReal<Real>>>
Real TransposedHelper
( const Matrix<Real>& d,
  const Matrix<Real>& NTrans,
  const Matrix<Real>& u,
        Matrix<Real>& v,
  const EnumCtrl<Real>& ctrl )
{
    DEBUG_ONLY(CSE cse("svp::gnr_enum::TransposedHelper"))
    const Int n = NTrans.Height();
    if( n != NTrans.Width() )
        LogicError("Expected height(N) = width(N)");

    Matrix<Real> S;
    Zeros( S, n+1, n );

    // The 'indices' vector is indexed differently from the 'r' from
    // Gama/Nguyen/Regev, and (r_0,r_1,r_2,...,r_n)=(0,1,2,...,n) becomes
    // (-1,0,1,...,n-1).
    Matrix<Int> indices;
    Zeros( indices, n+1, 1 );
    for( Int j=0; j<=n; ++j )
        indices(j) = j-1;

    // Note: We maintain the norms rather than their squares
    Matrix<Real> partialNorms;
    Zeros( partialNorms, n+1, 1 );

    Zeros( v, n, 1 );
    if( n == 0 )
        return Real(0);
    v(0) = Real(1);
    Real* vBuf = &v(0);

    Matrix<Real> centers;
    Zeros( centers, n, 1 );

    Matrix<Real> jumps;
    Zeros( jumps, n, 1 );

    Int lastNonzero = 0; // -1 if all indices are zero

    Int k=0;
    while( true )
    {
        Real diff = v(k)-centers(k);
        Real rho_k = lapack::SafeNorm( partialNorms(k+1), diff*d(k) );
        partialNorms(k) = rho_k;
        if( rho_k < u((n-1)-k) )
        {
            if( k == 0 )
            {
                // Success
                return rho_k;
            }
            else
            {
                // Move down the tree
                --k;
                indices(k) = Max(indices(k),indices(k+1));

                      Real* s = &S(0,k);
                const Real* nBuf = &NTrans(0,k);
                for( Int i=indices(k+1); i>=k+1; --i )
                    s[i] = s[i+1] + vBuf[i]*nBuf[i];

                centers(k) = -S(k+1,k);
                vBuf[k] = Round(centers(k));
                jumps(k) = Real(1);
            }
        }
        else
        {
            // Move up the tree
            ++k;
            if( k == n )
                return 2*u(n-1)+1; // An arbitrary value > than u(n-1)
            indices(k) = k; // indicate that (i,j) are not synchronized
            if( k >= lastNonzero )
            {
                if( ctrl.innerProgress )
                    Output("lastNonzero: ",k);
                lastNonzero = k;
                vBuf[k] += Real(1);
            }
            else
            {
                if( vBuf[k] > centers(k) )
                    vBuf[k] -= jumps(k);
                else
                    vBuf[k] += jumps(k);
                jumps(k) += Real(1);
            }
        }
    }
}

// TODO: Complex version here

} // namespace gnr_enum

template<typename F>
Base<F> GNREnumeration
( const Matrix<Base<F>>& d,
  const Matrix<F>& N,
  const Matrix<Base<F>>& u,
        Matrix<F>& v,
  const EnumCtrl<Base<F>>& ctrl )
{
    DEBUG_ONLY(CSE cse("svp::GNREnumeration"))
    if( ctrl.explicitTranspose )
    {
        Matrix<F> NTrans;
        Transpose( N, NTrans );
        return gnr_enum::TransposedHelper( d, NTrans, u, v, ctrl );
    }
    else
    {
        return gnr_enum::Helper( d, N, u, v, ctrl );
    }
}

} // namespace svp

#define PROTO(F) \
  template Base<F> svp::GNREnumeration \
  ( const Matrix<Base<F>>& d, \
    const Matrix<F>& N, \
    const Matrix<Base<F>>& u, \
          Matrix<F>& v, \
    const EnumCtrl<Base<F>>& ctrl );

#define EL_NO_INT_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#define EL_NO_COMPLEX_PROTO
#include <El/macros/Instantiate.h>

} // namespace El
