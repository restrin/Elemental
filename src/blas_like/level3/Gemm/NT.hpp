/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/

namespace El {
namespace gemm {

// Normal Transpose Gemm that avoids communicating the matrix A
template<typename T>
void SUMMA_NTA
( Orientation orientB,
  T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
        AbstractDistMatrix<T>& CPre )
{
    EL_DEBUG_CSE
    const Int n = CPre.Width();
    const Int bsize = Blocksize();
    const Grid& g = APre.Grid();
    const bool conjugate = ( orientB == ADJOINT );

    DistMatrixReadProxy<T,T,MC,MR> AProx( APre );
    DistMatrixReadProxy<T,T,MC,MR> BProx( BPre );
    DistMatrixReadWriteProxy<T,T,MC,MR> CProx( CPre );
    auto& A = AProx.GetLocked();
    auto& B = BProx.GetLocked();
    auto& C = CProx.Get();

    // Temporary distributions
    DistMatrix<T,MR,STAR> B1Trans_MR_STAR(g);
    DistMatrix<T,MC,STAR> D1_MC_STAR(g);

    B1Trans_MR_STAR.AlignWith( A );
    D1_MC_STAR.AlignWith( A );

    for( Int k=0; k<n; k+=bsize )
    {
        const Int nb = Min(bsize,n-k);
        auto B1 = B( IR(k,k+nb), ALL        );
        auto C1 = C( ALL,        IR(k,k+nb) );

        // C1[MC,*] := alpha A[MC,MR] (B1^[T/H])[MR,*]
        Transpose( B1, B1Trans_MR_STAR, conjugate );
        LocalGemm( NORMAL, NORMAL, alpha, A, B1Trans_MR_STAR, D1_MC_STAR );

        // C1[MC,MR] += scattered result of D1[MC,*] summed over grid rows
        AxpyContract( T(1), D1_MC_STAR, C1 );
    }
}

// Normal Transpose Gemm that avoids communicating the matrix B
template<typename T>
void SUMMA_NTB
( Orientation orientB,
  T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
        AbstractDistMatrix<T>& CPre )
{
    EL_DEBUG_CSE
    const Int m = CPre.Height();
    const Int bsize = Blocksize();
    const Grid& g = APre.Grid();

    DistMatrixReadProxy<T,T,MC,MR> AProx( APre );
    DistMatrixReadProxy<T,T,MC,MR> BProx( BPre );
    DistMatrixReadWriteProxy<T,T,MC,MR> CProx( CPre );
    auto& A = AProx.GetLocked();
    auto& B = BProx.GetLocked();
    auto& C = CProx.Get();

    // Temporary distributions
    DistMatrix<T,MR,STAR> A1Trans_MR_STAR(g);
    DistMatrix<T,STAR,MC> D1_STAR_MC(g);
    DistMatrix<T,MR,MC> D1_MR_MC(g);

    A1Trans_MR_STAR.AlignWith( B );
    D1_STAR_MC.AlignWith( B );

    for( Int k=0; k<m; k+=bsize )
    {
        const Int nb = Min(bsize,m-k);
        auto A1 = A( IR(k,k+nb), ALL );
        auto C1 = C( IR(k,k+nb), ALL );

        // D1[*,MC] := alpha A1[*,MR] (B[MC,MR])^T
        //           = alpha (A1^T)[MR,*] (B^T)[MR,MC]
        Transpose( A1, A1Trans_MR_STAR );
        LocalGemm( TRANSPOSE, orientB, alpha, A1Trans_MR_STAR, B, D1_STAR_MC );

        // C1[MC,MR] += scattered & transposed D1[*,MC] summed over grid rows
        Contract( D1_STAR_MC, D1_MR_MC );
        Axpy( T(1), D1_MR_MC, C1 );
    }
}

// Normal Transpose Gemm that avoids communicating the matrix C
template<typename T>
void SUMMA_NTC
( Orientation orientB,
  T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
        AbstractDistMatrix<T>& CPre )
{
    EL_DEBUG_CSE
    const Int sumDim = APre.Width();
    const Int bsize = Blocksize();
    const Grid& g = APre.Grid();
    const bool conjugate = ( orientB == ADJOINT );

    DistMatrixReadProxy<T,T,MC,MR> AProx( APre );
    DistMatrixReadProxy<T,T,MC,MR> BProx( BPre );
    DistMatrixReadWriteProxy<T,T,MC,MR> CProx( CPre );
    auto& A = AProx.GetLocked();
    auto& B = BProx.GetLocked();
    auto& C = CProx.Get();

    // Temporary distributions
    DistMatrix<T,MC,STAR> A1_MC_STAR(g);
    DistMatrix<T,VR,STAR> B1_VR_STAR(g);
    DistMatrix<T,STAR,MR> B1Trans_STAR_MR(g);

    A1_MC_STAR.AlignWith( C );
    B1_VR_STAR.AlignWith( C );
    B1Trans_STAR_MR.AlignWith( C );

    for( Int k=0; k<sumDim; k+=bsize )
    {
        const Int nb = Min(bsize,sumDim-k);
        auto A1 = A( ALL, IR(k,k+nb) );
        auto B1 = B( ALL, IR(k,k+nb) );

        A1_MC_STAR = A1;
        B1_VR_STAR = B1;
        Transpose( B1_VR_STAR, B1Trans_STAR_MR, conjugate );

        // C[MC,MR] += alpha A1[MC,*] (B1[MR,*])^T
        LocalGemm
        ( NORMAL, NORMAL, alpha, A1_MC_STAR, B1Trans_STAR_MR, T(1), C );
    }
}

// Normal Transpose Gemm for panel-panel dot products
//
// Use summations of local multiplications from a 1D distribution of A and B
// to update blockSize x blockSize submatrices of C
//
template<typename T>
void SUMMA_NTDot
( Orientation orientB,
  T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
        AbstractDistMatrix<T>& CPre,
  Int blockSize=2000 )
{
    EL_DEBUG_CSE
    const Int m = CPre.Height();
    const Int n = CPre.Width();
    const Grid& g = APre.Grid();

    DistMatrixReadProxy<T,T,STAR,VC> AProx( APre );
    auto& A = AProx.GetLocked();

    ElementalProxyCtrl BCtrl;
    BCtrl.rowConstrain = true;
    BCtrl.rowAlign = A.RowAlign();
    DistMatrixReadProxy<T,T,STAR,VC> BProx( BPre, BCtrl );
    auto& B = BProx.GetLocked();

    DistMatrixReadWriteProxy<T,T,MC,MR> CProx( CPre );
    auto& C = CProx.Get();

    DistMatrix<T,STAR,STAR> C11_STAR_STAR(g);
    for( Int kOuter=0; kOuter<m; kOuter+=blockSize )
    {
        const Int nbOuter = Min(blockSize,m-kOuter);
        const Range<Int> indOuter( kOuter, kOuter+nbOuter );

        auto A1 = A( indOuter, ALL );

        for( Int kInner=0; kInner<n; kInner+=blockSize )
        {
            const Int nbInner = Min(blockSize,n-kInner);
            const Range<Int> indInner( kInner, kInner+nbInner );

            auto B1  = B( indInner, ALL );
            auto C11 = C( indOuter, indInner );

            LocalGemm( NORMAL, orientB, alpha, A1, B1, C11_STAR_STAR );
            AxpyContract( T(1), C11_STAR_STAR, C11 );
        }
    }
}

template<typename T>
void SUMMA_NT
( Orientation orientB,
  T alpha,
  const AbstractDistMatrix<T>& A,
  const AbstractDistMatrix<T>& B,
        AbstractDistMatrix<T>& C,
  GemmAlgorithm alg=GEMM_DEFAULT )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      AssertSameGrids( A, B, C );
      if( orientB == NORMAL )
          LogicError("B must be (Conjugate)Transposed");
      if( A.Height() != C.Height() ||
          B.Height() != C.Width() ||
          A.Width() != B.Width() )
          LogicError
          ("Nonconformal matrices:\n",
           DimsString(A,"A"),"\n",
           DimsString(B,"B"),"\n",
           DimsString(C,"C"));
    )

    const Int m = C.Height();
    const Int n = C.Width();
    const Int sumDim = A.Width();
    const double weightTowardsC = 2.;
    const double weightAwayFromDot = 10.;

    // TODO(poulson): Make this tunable
    const Int blockSizeDot = 2000;

    switch( alg )
    {
    case GEMM_DEFAULT:
        if( weightAwayFromDot*m <= sumDim && weightAwayFromDot*n <= sumDim )
            SUMMA_NTDot( orientB, alpha, A, B, C, blockSizeDot );
        else if( m <= n && weightTowardsC*m <= sumDim )
            SUMMA_NTB( orientB, alpha, A, B, C );
        else if( n <= m && weightTowardsC*n <= sumDim )
            SUMMA_NTA( orientB, alpha, A, B, C );
        else
            SUMMA_NTC( orientB, alpha, A, B, C );
        break;
    case GEMM_SUMMA_A: SUMMA_NTA( orientB, alpha, A, B, C ); break;
    case GEMM_SUMMA_B: SUMMA_NTB( orientB, alpha, A, B, C ); break;
    case GEMM_SUMMA_C: SUMMA_NTC( orientB, alpha, A, B, C ); break;
    case GEMM_SUMMA_DOT: SUMMA_NTDot( orientB, alpha, A, B, C ); break;
    default: LogicError("Unsupported Gemm option");
    }
}

} // namespace gemm
} // namespace El
