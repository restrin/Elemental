/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_HESSENBERG_FORMQ_HPP
#define EL_HESSENBERG_FORMQ_HPP

namespace El {
namespace hessenberg {

template<typename F>
void FormQ
( UpperOrLower uplo,
  const Matrix<F>& A,
  const Matrix<F>& householderScalars,
        Matrix<F>& Q )
{
    EL_DEBUG_CSE
    // TODO: Make this smarter
    const Int n = A.Height();
    Identity( Q, n, n );
    ApplyQ( LEFT, uplo, NORMAL, A, householderScalars, Q );
}

template<typename F>
void FormQ
( UpperOrLower uplo,
  const AbstractDistMatrix<F>& A,
  const AbstractDistMatrix<F>& householderScalars,
        AbstractDistMatrix<F>& Q )
{
    EL_DEBUG_CSE
    // TODO: Make this smarter
    const Int n = A.Height();
    Identity( Q, n, n );
    ApplyQ( LEFT, uplo, NORMAL, A, householderScalars, Q );
}

} // namespace hessenberg
} // namespace El

#endif // ifndef EL_HESSENBERG_FORMQ_HPP
