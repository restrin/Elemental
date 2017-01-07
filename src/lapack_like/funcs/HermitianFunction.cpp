/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

namespace El {

// Modify the eigenvalues of A with the real-valued function f, which will
// therefore result in a Hermitian matrix, which we store in-place.

template<typename Field>
void HermitianFunction
( UpperOrLower uplo,
  Matrix<Field>& A,
  function<Base<Field>(const Base<Field>&)> func )
{
    EL_DEBUG_CSE
    if( A.Height() != A.Width() )
        LogicError("Hermitian matrices must be square");
    typedef Base<Field> Real;

    // Get the EVD of A
    Matrix<Real> w;
    Matrix<Field> Z;
    HermitianEig( uplo, A, w, Z );

    // Replace w with f(w)
    EntrywiseMap( w, func );

    // A := Z f(Omega) Z^H
    HermitianFromEVD( uplo, A, w, Z );
}

template<typename Field>
void HermitianFunction
( UpperOrLower uplo,
  AbstractDistMatrix<Field>& APre,
  function<Base<Field>(const Base<Field>&)> func )
{
    EL_DEBUG_CSE

    DistMatrixReadWriteProxy<Field,Field,MC,MR> AProx( APre );
    auto& A = AProx.Get();

    if( A.Height() != A.Width() )
        LogicError("Hermitian matrices must be square");
    typedef Base<Field> Real;

    // Get the EVD of A
    const Grid& g = A.Grid();
    DistMatrix<Real,VR,STAR> w(g);
    DistMatrix<Field> Z(g);
    HermitianEig( uplo, A, w, Z );

    // Replace w with f(w)
    EntrywiseMap( w, func );

    // A := Z f(Omega) Z^H
    HermitianFromEVD( uplo, A, w, Z );
}

// Modify the eigenvalues of A with the complex-valued function f, which will
// therefore result in a normal (in general, non-Hermitian) matrix, which we
// store in-place. At some point a version will be written which takes a real
// symmetric matrix as input and produces a complex normal matrix.

template<typename Real>
void HermitianFunction
( UpperOrLower uplo,
  Matrix<Complex<Real>>& A,
  function<Complex<Real>(const Real&)> func )
{
    EL_DEBUG_CSE
    if( A.Height() != A.Width() )
        LogicError("Hermitian matrices must be square");
    typedef Complex<Real> C;

    // Get the EVD of A
    Matrix<Real> w;
    Matrix<C> Z;
    HermitianEig( uplo, A, w, Z );

    // Form f(w)
    const Int n = w.Height();
    Matrix<C> fw( n, 1 );
    for( Int i=0; i<n; ++i )
        fw(i) = func(w(i));

    // A := Z f(Omega) Z^H
    NormalFromEVD( A, fw, Z );
}

template<typename Real>
void HermitianFunction
( UpperOrLower uplo,
  AbstractDistMatrix<Complex<Real>>& APre,
  function<Complex<Real>(const Real&)> func )
{
    EL_DEBUG_CSE
    typedef Complex<Real> C;

    DistMatrixReadWriteProxy<C,C,MC,MR> AProx( APre );
    auto& A = AProx.Get();

    if( A.Height() != A.Width() )
        LogicError("Hermitian matrices must be square");

    // Get the EVD of A
    const Grid& g = A.Grid();
    DistMatrix<Real,VR,STAR> w(g);
    DistMatrix<C> Z(g);
    HermitianEig( uplo, A, w, Z );

    // Form f(w)
    DistMatrix<C,VR,STAR> fw(g);
    fw.AlignWith( w.DistData() );
    fw.Resize( w.Height(), 1 );
    const Int numLocalEigs = w.LocalHeight();
    for( Int iLoc=0; iLoc<numLocalEigs; ++iLoc )
    {
        const Real omega = w.GetLocal(iLoc,0);
        fw.SetLocal(iLoc,0,func(omega));
    }

    // A := Z f(Omega) Z^H
    NormalFromEVD( A, fw, Z );
}

#define PROTO_COMPLEX(Field) \
  template void HermitianFunction \
  ( UpperOrLower uplo, \
    Matrix<Field>& A, \
    function<Base<Field>(const Base<Field>&)> func ); \
  template void HermitianFunction \
  ( UpperOrLower uplo, \
    AbstractDistMatrix<Field>& A, \
    function<Base<Field>(const Base<Field>&)> func );

#define PROTO_REAL(Real) \
  PROTO_COMPLEX(Real) \
  template void HermitianFunction \
  ( UpperOrLower uplo, \
    Matrix<Complex<Real>>& A, \
    function<Complex<Real>(const Real&)> func ); \
  template void HermitianFunction \
  ( UpperOrLower uplo, \
    AbstractDistMatrix<Complex<Real>>& A, \
    function<Complex<Real>(const Real&)> func );

#define EL_NO_INT_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El
