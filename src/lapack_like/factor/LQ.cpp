/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

#include "./LQ/ApplyQ.hpp"
#include "./LQ/Householder.hpp"

#include "./LQ/SolveAfter.hpp"

#include "./LQ/Explicit.hpp"

namespace El {

template<typename F> 
void LQ( Matrix<F>& A, Matrix<F>& t, Matrix<Base<F>>& d )
{
    DEBUG_ONLY(CSE cse("LQ"))
    lq::Householder( A, t, d );
}

template<typename F> 
void LQ
( ElementalMatrix<F>& A, ElementalMatrix<F>& t, 
  ElementalMatrix<Base<F>>& d )
{
    DEBUG_ONLY(CSE cse("LQ"))
    lq::Householder( A, t, d );
}

// Variants which perform (Businger-Golub) row-pivoting
// ====================================================
// TODO

#define PROTO(F) \
  template void LQ( Matrix<F>& A, Matrix<F>& t, Matrix<Base<F>>& d ); \
  template void LQ \
  ( ElementalMatrix<F>& A, \
    ElementalMatrix<F>& t, ElementalMatrix<Base<F>>& d ); \
  template void lq::ApplyQ \
  ( LeftOrRight side, Orientation orientation, \
    const Matrix<F>& A, const Matrix<F>& t, \
    const Matrix<Base<F>>& d, Matrix<F>& B ); \
  template void lq::ApplyQ \
  ( LeftOrRight side, Orientation orientation, \
    const ElementalMatrix<F>& A, const ElementalMatrix<F>& t, \
    const ElementalMatrix<Base<F>>& d, ElementalMatrix<F>& B ); \
  template void lq::SolveAfter \
  ( Orientation orientation, \
    const Matrix<F>& A, const Matrix<F>& t, \
    const Matrix<Base<F>>& d, const Matrix<F>& B, \
          Matrix<F>& X ); \
  template void lq::SolveAfter \
  ( Orientation orientation, \
    const ElementalMatrix<F>& A, const ElementalMatrix<F>& t, \
    const ElementalMatrix<Base<F>>& d, const ElementalMatrix<F>& B, \
          ElementalMatrix<F>& X ); \
  template void lq::Explicit( Matrix<F>& L, Matrix<F>& A ); \
  template void lq::Explicit \
  ( ElementalMatrix<F>& L, ElementalMatrix<F>& A ); \
  template void lq::ExplicitTriang( Matrix<F>& A ); \
  template void lq::ExplicitTriang( ElementalMatrix<F>& A ); \
  template void lq::ExplicitUnitary( Matrix<F>& A ); \
  template void lq::ExplicitUnitary( ElementalMatrix<F>& A );

#define EL_NO_INT_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El
