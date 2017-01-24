/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

#include "../util.hpp"

namespace El {
namespace pdco {

// Naive initialization for now. Will likely add analytic center
// initialization later.
// TODO: Accept x0, y0, z0
// TODO: Make this less ugly (maybe there's some syntatic sugar?)
// TODO: Treat z1,z2 as sparse vectors

template<typename Real>
void Initialize
(       Matrix<Real>& x,
        Matrix<Real>& y,
        Matrix<Real>& z1,
        Matrix<Real>& z2,
  const Matrix<Real>& bl,
  const Matrix<Real>& bu,
  const vector<Int>& ixSetLow,
  const vector<Int>& ixSetUpp,
  const vector<Int>& ixSetFix,
  const Real& x0min,
  const Real& z0min,
  const Int& m,
  const Int& n,
  bool print )
{
    EL_DEBUG_CSE

    if( print )
        Output("  Using naive initialization");

    // Initialize z1, z2
    Ones(z1, ixSetLow.size(), 1);
    z1 *= z0min;
    Ones(z2, ixSetUpp.size(), 1);
    z2 *= z0min;

    Int fixSize = ixSetFix.size();
    Int uppSize = ixSetUpp.size();
    Int lowSize = ixSetLow.size();

    //Initialize x, y
    Zeros(x, n, 1);
    Zeros(y, m, 1);

    Int ctrLow = 0;
    Int ctrUpp = 0;
    Int ctrFix = 0;

    for( Int i = 0; i < n; i++ )
    {
        if( fixSize> 0 && i == ixSetFix[ctrFix] )
        {
            x(i, 0) = bl(i,0);
            ctrFix++;
            continue;
        }
        if( lowSize > 0 && uppSize > 0
            && i == ixSetLow[ctrLow] && i == ixSetUpp[ctrUpp] )
        {
            Real val = Real(0.5)*(bl.Get(i,0) + bu.Get(i,0));
            x.Set(i, 0, val);
            ctrLow++;
            ctrUpp++;
            continue;
        }
        if( lowSize > 0 && i == ixSetLow[ctrLow] )
        {
            x.Set(i, 0, bl.Get(i,0) + x0min);
            ctrLow++;
            continue;
        }
        if( uppSize > 0 && i == ixSetUpp[ctrUpp] )
        {
            x.Set(i, 0, bu.Get(i,0) - x0min);
            ctrUpp++;
            continue;
        }
    }
}

// Ensure initialized variables are valid
template<typename Real>
void CheckVariableInit
( const Matrix<Real>& x,
  const Matrix<Real>& y,
  const Matrix<Real>& z,
  const Matrix<Real>& bl,
  const Matrix<Real>& bu,
  const vector<Int>& ixSetLow,
  const vector<Int>& ixSetUpp,
  const vector<Int>& ixSetFix )
{
    // Check if any variables are not initialized
    if( x.Height() <= 0 || y.Height() <= 0 || z.Height() <= 0 )
    {
        RuntimeError("Need to initialize all variables!");
    }

    // Ensure variables satisfy bounds
    for( Int i = 0; i < ixSetLow.size(); i++ )
    {
        if( x(i,0) <= bl(i,0) )
        {
            cout << "Below: i=" << i << " " << x(i,0) << " " << bl(i,0) << endl;
            RuntimeError("x must be strictly interior to the bounds!");
        }
    }
    for( Int i = 0; i < ixSetUpp.size(); i++ )
    {
        if( x(i,0) >= bu(i,0) )
        {
            cout << "Above: i=" << i << " " << x(i,0) << " " << bu(i,0) << endl;
            RuntimeError("x must be strictly interior to the bounds!");
        }
    }
    for( Int i = 0; i < ixSetFix.size(); i++ )
    {
        if( x(i,0) == bl(i,0) )
        {
            cout << "Fixed: i=" << i << " " << x(i,0) << " " << bl(i,0) << endl;
            RuntimeError("x must be fixed!");
        }
    }
}

#define PROTO(Real) \
  template void Initialize \
  (       Matrix<Real>& x, \
          Matrix<Real>& y, \
          Matrix<Real>& z1, \
          Matrix<Real>& z2, \
    const Matrix<Real>& bl, \
    const Matrix<Real>& bu, \
    const vector<Int>& ixSetLow, \
    const vector<Int>& ixSetUpp, \
    const vector<Int>& ixSetFix, \
    const Real& x0min, \
    const Real& z0min, \
    const Int& m, \
    const Int& n, \
    bool print ); \
  template void CheckVariableInit \
  ( const Matrix<Real>& x, \
    const Matrix<Real>& y, \
    const Matrix<Real>& z, \
    const Matrix<Real>& bl, \
    const Matrix<Real>& bu, \
    const vector<Int>& ixSetLow, \
    const vector<Int>& ixSetUpp, \
    const vector<Int>& ixSetFix );

#define EL_NO_INT_PROTO
#define EL_NO_COMPLEX_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include "El/macros/Instantiate.h"

} // namespace pdco
} // namespace El
