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

// This function is used to obtain the index sets which
// classify the primal variables based on whether they
// are bounded above, below or are fixed.
template<typename Real>
void ClassifyBounds
( const Matrix<Real>& bl,
  const Matrix<Real>& bu,
        vector<Int>& ixSetLow,
        vector<Int>& ixSetUpp,
        vector<Int>& ixSetFix,
  bool print )
{
    EL_DEBUG_CSE
    
    Int n = bl.Height();
    Int ctrLow = 0;
    Int ctrUpp = 0;
    Int ctrFix = 0;
    
    for( Int i=0; i < n; i++ )
    {
        // bl > bu?
        if( bl.Get(i,0) > bu.Get(i,0) )
            RuntimeError("Lower bound greater than upper bound at i=", i);

        // Fixed variable?
        // TODO: Checking equality may be dangerous
        //       Should change to tolerance?
        if( bl.Get(i,0) == bu.Get(i,0) )
        {
            ixSetFix.push_back(i);
            ctrFix++;
            continue;
        }
        // Lower bounded?
        if( limits::IsFinite(bl.Get(i,0)) )
        {
            ixSetLow.push_back(i);
            ctrLow++;
        }
        // Upper bounded?
        if( limits::IsFinite(bu.Get(i,0)) )
        {
            ixSetUpp.push_back(i);
            ctrUpp++;
        }
    }

    if( print )
    {
        Output("Bounds:");
        Output("  Finite bl: ", ctrLow);
        Output("  Finite bu: ", ctrUpp);
        Output("  Fixed    : ", ctrFix);
    }
}

#define PROTO(Real) \
    template void ClassifyBounds \
    ( const Matrix<Real>& bl, \
      const Matrix<Real>& bu, \
            vector<Int>& ixSetLow, \
            vector<Int>& ixSetUpp, \
            vector<Int>& ixSetFix, \
      bool print );

#define EL_NO_INT_PROTO
#define EL_NO_COMPLEX_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include "El/macros/Instantiate.h"

} // namespace pdco
} // namespace El
