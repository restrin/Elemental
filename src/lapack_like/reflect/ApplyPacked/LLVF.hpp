/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_APPLYPACKEDREFLECTORS_LLVF_HPP
#define EL_APPLYPACKEDREFLECTORS_LLVF_HPP

namespace El {

Timer LLVFTimer, GemmTimer, HerkTimer, TrsmTimer;

namespace apply_packed_reflectors {

//
// Since applying Householder transforms from vectors stored left-to-right
// implies that we will be forming a generalization of
//
//   (I - tau_1 u_1 u_1^H) (I - tau_0 u_0 u_0^H) = 
//   I - tau_0 u_0 u_0^H - tau_1 u_1 u_1^H + (tau_0 tau_1 u_1^H u_0) u_1 u_0^H =
//   I - [ u_0, u_1 ] [  tau_0,                 0     ] [ u_0^H ]
//                    [ -tau_0 tau_1 u_1^H u_0, tau_1 ] [ u_1^H ],
//
// which has a lower-triangular center matrix, say S, we will form S as 
// the inverse of a matrix T, which can easily be formed as
// 
//   tril(T) = tril( U^H U ),  diag(T) = 1/t or 1/conj(t),
//
// where U is the matrix of Householder vectors and t is the vector of scalars.
//

template<typename F> 
inline void
LLVF
( Conjugation conjugation,
  Int offset, 
  const Matrix<F>& H,
  const Matrix<F>& t,
        Matrix<F>& A )
{
    LLVFTimer.Reset();
    GemmTimer.Reset();
    HerkTimer.Reset();
    TrsmTimer.Reset();

    LLVFTimer.Start();

    DEBUG_ONLY(
      CSE cse("apply_packed_reflectors::LLVF");
      if( H.Height() != A.Height() )
          LogicError("A and H must be the same height");
    )
    const Int m = H.Height();
    const Int diagLength = H.DiagonalLength(offset);
    DEBUG_ONLY(
      if( t.Height() != diagLength )
          LogicError("t must be the same length as H's offset diag.");
    )
    Matrix<F> HPanCopy, SInv, Z;

    const Int iOff = ( offset>=0 ? 0      : -offset );
    const Int jOff = ( offset>=0 ? offset : 0       );

    const Int bsize = Blocksize();
    for( Int k=0; k<diagLength; k+=bsize )
    {
        const Int nb = Min(bsize,diagLength-k);
        const Int ki = k+iOff;
        const Int kj = k+jOff;

        auto HPan = H( IR(ki,m),   IR(kj,kj+nb) );
        auto ABot = A( IR(ki,m),   ALL          );
        auto t1   = t( IR(k,k+nb), ALL          );

        HPanCopy = HPan;
        MakeTrapezoidal( LOWER, HPanCopy );
        FillDiagonal( HPanCopy, F(1) );

        HerkTimer.Start();
        Herk( LOWER, ADJOINT, Base<F>(1), HPanCopy, SInv );
        HerkTimer.Stop();
        FixDiagonal( conjugation, t1, SInv );

        GemmTimer.Start();
        Gemm( ADJOINT, NORMAL, F(1), HPanCopy, ABot, Z );
        GemmTimer.Stop();
        TrsmTimer.Start();
        Trsm( LEFT, LOWER, NORMAL, NON_UNIT, F(1), SInv, Z );
        TrsmTimer.Stop();
        GemmTimer.Start();
        Gemm( NORMAL, NORMAL, F(-1), HPanCopy, Z, F(1), ABot );
        GemmTimer.Stop();
    }
    LLVFTimer.Stop();
//    Output("LLVF Total Time: ", LLVFTimer.Total());
//    Output("  GemmTimer: ", GemmTimer.Total());
//    Output("  HerkTimer: ", HerkTimer.Total());
//    Output("  TrsmTimer: ", TrsmTimer.Total());
}

template<typename F> 
inline void
LLVF
( Conjugation conjugation,
  Int offset, 
  const ElementalMatrix<F>& HPre,
  const ElementalMatrix<F>& tPre, 
        ElementalMatrix<F>& APre )
{
    DEBUG_ONLY(
      CSE cse("apply_packed_reflectors::LLVF");
      if( HPre.Height() != APre.Height() )
          LogicError("A and H must be the same height");
      AssertSameGrids( HPre, tPre, APre );
    )

    DistMatrixReadProxy<F,F,MC,MR  > HProx( HPre );
    DistMatrixReadProxy<F,F,MC,STAR> tProx( tPre );
    DistMatrixReadWriteProxy<F,F,MC,MR  > AProx( APre );
    auto& H = HProx.GetLocked();
    auto& t = tProx.GetLocked();
    auto& A = AProx.Get();

    const Int m = H.Height();
    const Int diagLength = H.DiagonalLength(offset);
    DEBUG_ONLY(
      if( t.Height() != diagLength )
          LogicError("t must be the same length as H's offset diag.");
    )
    const Grid& g = H.Grid();
    DistMatrix<F> HPanCopy(g);
    DistMatrix<F,VC,  STAR> HPan_VC_STAR(g);
    DistMatrix<F,MC,  STAR> HPan_MC_STAR(g);
    DistMatrix<F,STAR,STAR> t1_STAR_STAR(g), SInv_STAR_STAR(g);
    DistMatrix<F,STAR,MR  > Z_STAR_MR(g);
    DistMatrix<F,STAR,VR  > Z_STAR_VR(g);

    const Int iOff = ( offset>=0 ? 0      : -offset );
    const Int jOff = ( offset>=0 ? offset : 0       );

    const Int bsize = Blocksize();
    for( Int k=0; k<diagLength; k+=bsize )
    {
        const Int nb = Min(bsize,diagLength-k);
        const Int ki = k+iOff;
        const Int kj = k+jOff;

        auto HPan = H( IR(ki,m),   IR(kj,kj+nb) );
        auto ABot = A( IR(ki,m),   ALL          );
        auto t1   = t( IR(k,k+nb), ALL          );

        HPanCopy = HPan;
        MakeTrapezoidal( LOWER, HPanCopy );
        FillDiagonal( HPanCopy, F(1) );

        HPan_VC_STAR = HPanCopy;
        Zeros( SInv_STAR_STAR, nb, nb );
        Herk
        ( LOWER, ADJOINT, 
          Base<F>(1), HPan_VC_STAR.LockedMatrix(),
          Base<F>(0), SInv_STAR_STAR.Matrix() );     
        El::AllReduce( SInv_STAR_STAR, HPan_VC_STAR.ColComm() );
        t1_STAR_STAR = t1;
        FixDiagonal( conjugation, t1_STAR_STAR, SInv_STAR_STAR );

        HPan_MC_STAR.AlignWith( ABot );
        HPan_MC_STAR = HPanCopy;
        Z_STAR_MR.AlignWith( ABot );
        LocalGemm( ADJOINT, NORMAL, F(1), HPan_MC_STAR, ABot, Z_STAR_MR );
        Z_STAR_VR.AlignWith( ABot );
        Contract( Z_STAR_MR, Z_STAR_VR );
        
        LocalTrsm
        ( LEFT, LOWER, NORMAL, NON_UNIT, F(1), SInv_STAR_STAR, Z_STAR_VR );

        Z_STAR_MR = Z_STAR_VR;
        LocalGemm( NORMAL, NORMAL, F(-1), HPan_MC_STAR, Z_STAR_MR, F(1), ABot );
    }
}

} // namespace apply_packed_reflectors
} // namespace El

#endif // ifndef EL_APPLYPACKEDREFLECTORS_LLVF_HPP
