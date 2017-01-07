/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

namespace El {
namespace soc {

// sqrt(x) = [ eta_0; x_1/(2 eta_0) ],
// where eta_0 = sqrt(x_0 + sqrt(det(x))) / sqrt(2).

template<typename Real,
         typename/*=EnableIf<IsReal<Real>>*/>
void SquareRoot
( const Matrix<Real>& x,
        Matrix<Real>& xRoot,
  const Matrix<Int>& orders,
  const Matrix<Int>& firstInds )
{
    EL_DEBUG_CSE

    Matrix<Real> d;
    soc::Dets( x, d, orders, firstInds );
    cone::Broadcast( d, orders, firstInds );

    const Int height = x.Height();
    Zeros( xRoot, height, 1 );
    for( Int i=0; i<height; )
    {
        const Int order = orders(i);
        const Int firstInd = firstInds(i);
        EL_DEBUG_ONLY(
          if( i != firstInd )
              LogicError("Inconsistency in orders and firstInds");
        )

        const Real eta0 = Sqrt(x(i)+Sqrt(d(i)))/Sqrt(Real(2));
        xRoot(i) = eta0;
        for( Int k=1; k<order; ++k )
            xRoot(i+k) = x(i+k)/(2*eta0);

        i += order;
    }
}

template<typename Real,
         typename/*=EnableIf<IsReal<Real>>*/>
void SquareRoot
( const AbstractDistMatrix<Real>& xPre,
        AbstractDistMatrix<Real>& xRootPre,
  const AbstractDistMatrix<Int>& ordersPre,
  const AbstractDistMatrix<Int>& firstIndsPre,
  Int cutoff )
{
    EL_DEBUG_CSE
    AssertSameGrids( xPre, xRootPre, ordersPre, firstIndsPre );

    ElementalProxyCtrl ctrl;
    ctrl.colConstrain = true;
    ctrl.colAlign = 0;

    DistMatrixReadProxy<Real,Real,VC,STAR>
      xProx( xPre, ctrl );
    DistMatrixWriteProxy<Real,Real,VC,STAR>
      xRootProx( xRootPre, ctrl );
    DistMatrixReadProxy<Int,Int,VC,STAR>
      ordersProx( ordersPre, ctrl ),
      firstIndsProx( firstIndsPre, ctrl );
    auto& x = xProx.GetLocked();
    auto& xRoot = xRootProx.Get();
    auto& orders = ordersProx.GetLocked();
    auto& firstInds = firstIndsProx.GetLocked();

    const Real* xBuf = x.LockedBuffer();
    const Int* firstIndBuf = firstInds.LockedBuffer();

    DistMatrix<Real,VC,STAR> d(x.Grid());
    soc::Dets( x, d, orders, firstInds );
    cone::Broadcast( d, orders, firstInds );
    const Real* dBuf = d.LockedBuffer();

    auto roots = x;
    cone::Broadcast( roots, orders, firstInds );
    const Real* rootBuf = roots.LockedBuffer();

    const Int localHeight = x.LocalHeight();
    xRoot.SetGrid( x.Grid() );
    Zeros( xRoot, x.Height(), 1 );
    Real* xRootBuf = xRoot.Buffer();
    for( Int iLoc=0; iLoc<localHeight; ++iLoc )
    {
        const Int i = x.GlobalRow(iLoc);
        const Real x0 = rootBuf[iLoc];
        const Real det = dBuf[iLoc];
        const Real eta0 = Sqrt(x0+Sqrt(det))/Sqrt(Real(2));
        if( i == firstIndBuf[iLoc] )
            xRootBuf[iLoc] = eta0;
        else
            xRootBuf[iLoc] = xBuf[iLoc]/(2*eta0);
    }
}

template<typename Real,
         typename/*=EnableIf<IsReal<Real>>*/>
void SquareRoot
( const DistMultiVec<Real>& x,
        DistMultiVec<Real>& xRoot,
  const DistMultiVec<Int>& orders,
  const DistMultiVec<Int>& firstInds,
  Int cutoff )
{
    EL_DEBUG_CSE
    const Grid& grid = x.Grid();

    const Real* xBuf = x.LockedMatrix().LockedBuffer();
    const Int* firstIndBuf = firstInds.LockedMatrix().LockedBuffer();

    DistMultiVec<Real> d(grid);
    soc::Dets( x, d, orders, firstInds );
    cone::Broadcast( d, orders, firstInds );
    const Real* dBuf = d.LockedMatrix().LockedBuffer();

    auto roots = x;
    cone::Broadcast( roots, orders, firstInds );
    const Real* rootBuf = roots.LockedMatrix().LockedBuffer();

    const Int localHeight = x.LocalHeight();
    xRoot.SetGrid( grid );
    Zeros( xRoot, x.Height(), 1 );
    Real* xRootBuf = xRoot.Matrix().Buffer();
    for( Int iLoc=0; iLoc<localHeight; ++iLoc )
    {
        const Int i = x.GlobalRow(iLoc);
        const Real x0 = rootBuf[iLoc];
        const Real det = dBuf[iLoc];
        const Real eta0 = Sqrt(x0+Sqrt(det))/Sqrt(Real(2));
        if( i == firstIndBuf[iLoc] )
            xRootBuf[iLoc] = eta0;
        else
            xRootBuf[iLoc] = xBuf[iLoc]/(2*eta0);
    }
}

#define PROTO(Real) \
  template void SquareRoot \
  ( const Matrix<Real>& x, \
          Matrix<Real>& xRoot, \
    const Matrix<Int>& orders, \
    const Matrix<Int>& firstInds ); \
  template void SquareRoot \
  ( const AbstractDistMatrix<Real>& x, \
          AbstractDistMatrix<Real>& xRoot, \
    const AbstractDistMatrix<Int>& orders, \
    const AbstractDistMatrix<Int>& firstInds, \
    Int cutoff ); \
  template void SquareRoot \
  ( const DistMultiVec<Real>& x, \
          DistMultiVec<Real>& xRoot, \
    const DistMultiVec<Int>& orders, \
    const DistMultiVec<Int>& firstInds, \
    Int cutoff );

#define EL_NO_INT_PROTO
#define EL_NO_COMPLEX_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace soc
} // namespace El
