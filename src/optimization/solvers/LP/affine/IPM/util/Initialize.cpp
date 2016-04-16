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
namespace lp {
namespace affine {

//
// Despite the fact that the CVXOPT documentation [1] suggests a single-stage
// procedure for initializing (x,y,z,s), a post-processed two-stage procedure 
// is currently used by the code [2]:
//
// 1) Minimize || G x - h ||^2, s.t. A x = b  by solving
//
//    | 0 A^T G^T | |  x |   | 0 |
//    | A  0   0  | |  u | = | b |,
//    | G  0  -I  | | -s |   | h |
//
//   where 'u' is an unused dummy variable.
//
// 2) Minimize || z ||^2, s.t. A^T y + G^T z + c = 0 by solving
//
//    | 0 A^T G^T | | u |   | -c |
//    | A  0   0  | | y | = |  0 |,
//    | G  0  -I  | | z |   |  0 |
//
//    where 'u' is an unused dummy variable.
//
// 3) Set 
//
//      alpha_p := -min(s), and
//      alpha_d := -min(z).
//
//    Then shift s and z according to the rules:
//
//      s := ( alpha_p > -sqrt(eps)*Max(1,||s||_2) ? s + (1+alpha_p)e : s )
//      z := ( alpha_d > -sqrt(eps)*Max(1,||z||_2) ? z + (1+alpha_d)e : z ),
//
//    where 'eps' is the machine precision, 'e' is a vector of all ones 
//    (for more general conic optimization problems, it is the product of 
//    identity elements from the Jordan algebras whose squares yield the 
//    relevant cone.
//
//    Since the post-processing in step (3) has a large discontinuity as the 
//    minimum entry approaches sqrt(eps)*Max(1,||q||_2), we also provide
//    the ability to instead use an entrywise lower clip.
//
// [1] L. Vandenberghe
//     "The CVXOPT linear and quadratic cone program solvers"
//     <http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf>
//
// [2] L. Vandenberghe
//     CVXOPT's source file, "src/python/coneprog.py"
//     <https://github.com/cvxopt/cvxopt/blob/f3ca94fb997979a54b913f95b816132f7fd44820/src/python/coneprog.py>
//

template<typename Real>
void Initialize
( const Matrix<Real>& A,
  const Matrix<Real>& G,
  const Matrix<Real>& b,
  const Matrix<Real>& c,
  const Matrix<Real>& h,
        Matrix<Real>& x,
        Matrix<Real>& y,
        Matrix<Real>& z,
        Matrix<Real>& s,
  bool primalInit, bool dualInit, bool standardShift )
{
    DEBUG_ONLY(CSE cse("lp::affine::Initialize"))
    const Int m = A.Height();
    const Int n = A.Width();
    const Int k = G.Height();
    if( primalInit )
    {
        if( x.Height() != n || x.Width() != 1 )
            LogicError("x was of the wrong size");
        if( s.Height() != k || s.Width() != 1 )
            LogicError("s was of the wrong size");
    }
    if( dualInit )
    {
        if( y.Height() != m || y.Width() != 1 )
            LogicError("y was of the wrong size");
        if( z.Height() != k || z.Width() != 1 )
            LogicError("z was of the wrong size");
    }
    if( primalInit && dualInit )
    {
        // TODO: Perform a consistency check
        return;
    }

    // Form the KKT matrix
    // ===================
    Matrix<Real> J, ones;
    Ones( ones, k, 1 );
    KKT( A, G, ones, ones, J );

    // Factor the KKT matrix
    // =====================
    Matrix<Real> dSub;
    Permutation p;
    LDL( J, dSub, p, false );

    Matrix<Real> rc, rb, rh, rmu, u, d;
    Zeros( rmu, k, 1 );
    if( !primalInit )
    {
        // Minimize || G x - h ||^2, s.t. A x = b  by solving
        //
        //    | 0 A^T G^T | |  x |   | 0 |
        //    | A  0   0  | |  u | = | b |,
        //    | G  0  -I  | | -s |   | h |
        //
        //   where 'u' is an unused dummy variable.
        Zeros( rc, n, 1 );
        rb = b;
        rb *= -1;
        rh = h;
        rh *= -1;
        KKTRHS( rc, rb, rh, rmu, ones, d );
        ldl::SolveAfter( J, dSub, p, d, false );
        ExpandCoreSolution( m, n, k, d, x, u, s );
        s *= -1;
    }
    if( !dualInit )
    {
        // Minimize || z ||^2, s.t. A^T y + G^T z + c = 0 by solving
        //
        //    | 0 A^T G^T | | u |   | -c |
        //    | A  0   0  | | y | = |  0 |,
        //    | G  0  -I  | | z |   |  0 |
        //
        //    where 'u' is an unused dummy variable.
        rc = c;
        Zeros( rb, m, 1 );
        Zeros( rh, k, 1 );
        KKTRHS( rc, rb, rh, rmu, ones, d );
        ldl::SolveAfter( J, dSub, p, d, false );
        ExpandCoreSolution( m, n, k, d, u, y, z );
    }

    const Real epsilon = limits::Epsilon<Real>();
    const Real sNorm = Nrm2( s );
    const Real zNorm = Nrm2( z );
    const Real gammaPrimal = Sqrt(epsilon)*Max(sNorm,Real(1));
    const Real gammaDual   = Sqrt(epsilon)*Max(zNorm,Real(1));
    if( standardShift )
    {
        // alpha_p := min { alpha : s + alpha*e >= 0 }
        // -------------------------------------------
        const auto sMinPair = VectorMinLoc( s );
        const Real alphaPrimal = -sMinPair.value;
        if( alphaPrimal >= Real(0) && primalInit )
            RuntimeError("initialized s was non-positive");

        // alpha_d := min { alpha : z + alpha*e >= 0 }
        // -------------------------------------------
        const auto zMinPair = VectorMinLoc( z );
        const Real alphaDual = -zMinPair.value;
        if( alphaDual >= Real(0) && dualInit )
            RuntimeError("initialized z was non-positive");

        if( alphaPrimal >= -gammaPrimal )
            Shift( s, alphaPrimal+1 );
        if( alphaDual >= -gammaDual )
            Shift( z, alphaDual+1 );
    }
    else
    {
        LowerClip( s, gammaPrimal );
        LowerClip( z, gammaDual   );
    }
}

template<typename Real>
void Initialize
( const ElementalMatrix<Real>& A,
  const ElementalMatrix<Real>& G,
  const ElementalMatrix<Real>& b,
  const ElementalMatrix<Real>& c,
  const ElementalMatrix<Real>& h,
        ElementalMatrix<Real>& x,
        ElementalMatrix<Real>& y,
        ElementalMatrix<Real>& z,
        ElementalMatrix<Real>& s,
  bool primalInit, bool dualInit, bool standardShift )
{
    DEBUG_ONLY(CSE cse("lp::affine::Initialize"))
    const Int m = A.Height();
    const Int n = A.Width();
    const Int k = G.Height();
    const Grid& g = A.Grid();
    if( primalInit ) 
    {
        if( x.Height() != n || x.Width() != 1 )
            LogicError("x was of the wrong size");
        if( s.Height() != k || s.Width() != 1 )
            LogicError("s was of the wrong size");
    }
    if( dualInit )
    {
        if( y.Height() != m || y.Width() != 1 )
            LogicError("y was of the wrong size");
        if( z.Height() != k || z.Width() != 1 )
            LogicError("z was of the wrong size");
    }
    if( primalInit && dualInit )
    {
        // TODO: Perform a consistency check
        return;
    }

    // Form the KKT matrix
    // ===================
    DistMatrix<Real> J(g), ones(g);
    Ones( ones, k, 1 );
    KKT( A, G, ones, ones, J );

    // Factor the KKT matrix
    // =====================
    DistMatrix<Real> dSub(g);
    DistPermutation p(g);
    LDL( J, dSub, p, false );

    DistMatrix<Real> rc(g), rb(g), rh(g), rmu(g), d(g), u(g);
    Zeros( rmu, k, 1 );
    if( !primalInit )
    {
        // Minimize || G x - h ||^2, s.t. A x = b  by solving
        //
        //    | 0 A^T G^T | |  x |   | 0 |
        //    | A  0   0  | |  u | = | b |,
        //    | G  0  -I  | | -s |   | h |
        //
        //   where 'u' is an unused dummy variable.
        Zeros( rc, n, 1 );
        rb = b;
        rb *= -1;
        rh = h;
        rh *= -1;
        KKTRHS( rc, rb, rh, rmu, ones, d );
        ldl::SolveAfter( J, dSub, p, d, false );
        ExpandCoreSolution( m, n, k, d, x, u, s );
        s *= -1;
    }
    if( !dualInit )
    {
        // Minimize || z ||^2, s.t. A^T y + G^T z + c = 0 by solving
        //
        //    | 0 A^T G^T | | u |   | -c |
        //    | A  0   0  | | y | = |  0 |,
        //    | G  0  -I  | | z |   |  0 |
        //
        //    where 'u' is an unused dummy variable.
        rc = c;
        Zeros( rb, m, 1 );
        Zeros( rh, k, 1 );
        KKTRHS( rc, rb, rh, rmu, ones, d );
        ldl::SolveAfter( J, dSub, p, d, false );
        ExpandCoreSolution( m, n, k, d, u, y, z );
    }

    const Real epsilon = limits::Epsilon<Real>();
    const Real sNorm = Nrm2( s );
    const Real zNorm = Nrm2( z );
    const Real gammaPrimal = Sqrt(epsilon)*Max(sNorm,Real(1));
    const Real gammaDual   = Sqrt(epsilon)*Max(zNorm,Real(1));
    if( standardShift )
    {
        // alpha_p := min { alpha : s + alpha*e >= 0 }
        // -------------------------------------------
        const auto sMinPair = VectorMinLoc( s );
        const Real alphaPrimal = -sMinPair.value;
        if( alphaPrimal >= Real(0) && primalInit )
            RuntimeError("initialized s was non-positive");

        // alpha_d := min { alpha : z + alpha*e >= 0 }
        // -------------------------------------------
        const auto zMinPair = VectorMinLoc( z );
        const Real alphaDual = -zMinPair.value;
        if( alphaDual >= Real(0) && dualInit )
            RuntimeError("initialized z was non-positive");

        if( alphaPrimal >= -gammaPrimal )
            Shift( s, alphaPrimal+1 );
        if( alphaDual >= -gammaDual )
            Shift( z, alphaDual+1 );
    }
    else
    {
        LowerClip( s, gammaPrimal );
        LowerClip( z, gammaDual   );
    }
}

template<typename Real>
void Initialize
( const SparseMatrix<Real>& JStatic,
  const Matrix<Real>& regTmp,
  const Matrix<Real>& b,
  const Matrix<Real>& c,
  const Matrix<Real>& h,
        Matrix<Real>& x,
        Matrix<Real>& y,
        Matrix<Real>& z,
        Matrix<Real>& s,
  const vector<Int>& map,
  const vector<Int>& invMap, 
  const ldl::Separator& rootSep,
  const ldl::NodeInfo& info,
  bool primalInit, bool dualInit, bool standardShift,
  const RegSolveCtrl<Real>& solveCtrl )
{
    DEBUG_ONLY(CSE cse("lp::affine::Initialize"))
    qp::affine::Initialize
    ( JStatic, regTmp, b, c, h, x, y, z, s, map, invMap, rootSep, info,
      primalInit, dualInit, standardShift, solveCtrl );
}

template<typename Real>
void Initialize
( const DistSparseMatrix<Real>& JStatic,
  const DistMultiVec<Real>& regTmp,
  const DistMultiVec<Real>& b, 
  const DistMultiVec<Real>& c,
  const DistMultiVec<Real>& h,
        DistMultiVec<Real>& x,
        DistMultiVec<Real>& y,
        DistMultiVec<Real>& z,
        DistMultiVec<Real>& s,
  const DistMap& map,
  const DistMap& invMap, 
  const ldl::DistSeparator& rootSep,
  const ldl::DistNodeInfo& info,
        vector<Int>& mappedSources,
        vector<Int>& mappedTargets,
        vector<Int>& colOffs,
  bool primalInit, bool dualInit, bool standardShift, 
  const RegSolveCtrl<Real>& solveCtrl )
{
    DEBUG_ONLY(CSE cse("lp::affine::Initialize"))
    qp::affine::Initialize
    ( JStatic, regTmp, b, c, h, x, y, z, s,
      map, invMap, rootSep, info, mappedSources, mappedTargets, colOffs,
      primalInit, dualInit, standardShift, solveCtrl );
}

#define PROTO(Real) \
  template void Initialize \
  ( const Matrix<Real>& A, \
    const Matrix<Real>& G, \
    const Matrix<Real>& b, \
    const Matrix<Real>& c, \
    const Matrix<Real>& h, \
          Matrix<Real>& x, \
          Matrix<Real>& y, \
          Matrix<Real>& z, \
          Matrix<Real>& s, \
    bool primalInit, bool dualInit, bool standardShift ); \
  template void Initialize \
  ( const ElementalMatrix<Real>& A, \
    const ElementalMatrix<Real>& G, \
    const ElementalMatrix<Real>& b, \
    const ElementalMatrix<Real>& c, \
    const ElementalMatrix<Real>& h, \
          ElementalMatrix<Real>& x, \
          ElementalMatrix<Real>& y, \
          ElementalMatrix<Real>& z, \
          ElementalMatrix<Real>& s, \
    bool primalInit, bool dualInit, bool standardShift ); \
  template void Initialize \
  ( const SparseMatrix<Real>& JStatic, \
    const Matrix<Real>& regTmp, \
    const Matrix<Real>& b, \
    const Matrix<Real>& c, \
    const Matrix<Real>& h, \
          Matrix<Real>& x, \
          Matrix<Real>& y, \
          Matrix<Real>& z, \
          Matrix<Real>& s, \
    const vector<Int>& map, \
    const vector<Int>& invMap, \
    const ldl::Separator& rootSep, \
    const ldl::NodeInfo& info, \
    bool primalInit, bool dualInit, bool standardShift, \
    const RegSolveCtrl<Real>& solveCtrl ); \
  template void Initialize \
  ( const DistSparseMatrix<Real>& JStatic, \
    const DistMultiVec<Real>& regTmp, \
    const DistMultiVec<Real>& b, \
    const DistMultiVec<Real>& c, \
    const DistMultiVec<Real>& h, \
          DistMultiVec<Real>& x, \
          DistMultiVec<Real>& y, \
          DistMultiVec<Real>& z, \
          DistMultiVec<Real>& s, \
    const DistMap& map, \
    const DistMap& invMap, \
    const ldl::DistSeparator& rootSep, \
    const ldl::DistNodeInfo& info, \
          vector<Int>& mappedSources, \
          vector<Int>& mappedTargets, \
          vector<Int>& colOffs, \
    bool primalInit, bool dualInit, bool standardShift, \
    const RegSolveCtrl<Real>& solveCtrl );

#define EL_NO_INT_PROTO
#define EL_NO_COMPLEX_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace affine
} // namespace lp
} // namespace El
