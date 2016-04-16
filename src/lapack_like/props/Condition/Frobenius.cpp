/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

namespace El {

template<typename F> 
Base<F> FrobeniusCondition( const Matrix<F>& A )
{
    DEBUG_ONLY(CSE cse("FrobeniusCondition"))
    typedef Base<F> Real;
    Matrix<F> B( A );
    const Real oneNorm = FrobeniusNorm( B );
    try { Inverse( B ); }
    catch( SingularMatrixException& e )
    { return limits::Infinity<Real>(); }
    const Real oneNormInv = FrobeniusNorm( B );
    return oneNorm*oneNormInv;
}

template<typename F> 
Base<F> FrobeniusCondition( const ElementalMatrix<F>& A )
{
    DEBUG_ONLY(CSE cse("FrobeniusCondition"))
    typedef Base<F> Real;
    DistMatrix<F> B( A );
    const Real oneNorm = FrobeniusNorm( B );
    try { Inverse( B ); }
    catch( SingularMatrixException& e )
    { return limits::Infinity<Real>(); }
    const Real oneNormInv = FrobeniusNorm( B );
    return oneNorm*oneNormInv;
}

#define PROTO(F) \
  template Base<F> FrobeniusCondition( const Matrix<F>& A ); \
  template Base<F> FrobeniusCondition( const ElementalMatrix<F>& A );

#define EL_NO_INT_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El
