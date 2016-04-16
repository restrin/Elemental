/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

#include "./General/LUPartialPiv.hpp"

namespace El {

template<typename F> 
void Inverse( Matrix<F>& A )
{
    DEBUG_ONLY(CSE cse("Inverse"))
    inverse::LUPartialPiv( A );
}

template<typename F> 
void Inverse( ElementalMatrix<F>& A )
{
    DEBUG_ONLY(CSE cse("Inverse"))
    inverse::LUPartialPiv( A );
}

template<typename F>
void LocalInverse( DistMatrix<F,STAR,STAR>& A )
{
    DEBUG_ONLY(CSE cse("LocalInverse"))
    Inverse( A.Matrix() );
}

#define PROTO(F) \
  template void Inverse( Matrix<F>& A ); \
  template void Inverse( ElementalMatrix<F>& A ); \
  template void LocalInverse( DistMatrix<F,STAR,STAR>& A ); \
  template void inverse::AfterLUPartialPiv \
  ( Matrix<F>& A, const Permutation& P ); \
  template void inverse::AfterLUPartialPiv \
  ( ElementalMatrix<F>& A, const DistPermutation& P ); 

#define EL_NO_INT_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El
