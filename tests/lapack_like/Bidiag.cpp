/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>
using namespace El;

template<typename F> 
void TestCorrectness
( const DistMatrix<F>& A, 
  const DistMatrix<F,STAR,STAR>& tP,
  const DistMatrix<F,STAR,STAR>& tQ,
        DistMatrix<F>& AOrig,
  bool print,
  bool display )
{
    typedef Base<F> Real;
    const Grid& g = A.Grid();
    const Int m = AOrig.Height();
    const Int n = AOrig.Width();
    const Real eps = limits::Epsilon<Real>();
    const Real oneNormAOrig = OneNorm( AOrig );
    OutputFromRoot(g.Comm(),"Testing error...");
    PushIndent();

    // Grab the diagonal and superdiagonal of the bidiagonal matrix
    auto d = GetDiagonal( A, 0 );
    auto e = GetDiagonal( A, (m>=n ? 1 : -1) );

    // Zero B and then fill its bidiagonal
    DistMatrix<F> B(g);
    B.AlignWith( A );
    Zeros( B, m, n );
    SetDiagonal( B, d, 0  );
    SetDiagonal( B, e, (m>=n ? 1 : -1) );
    if( print )
        Print( B, "Bidiagonal" );
    if( display )
        Display( B, "Bidiagonal" );

    if( print || display )
    {
        DistMatrix<F> Q(g), P(g);
        Identity( Q, m, m );
        Identity( P, n, n );
        bidiag::ApplyQ( LEFT,  NORMAL, A, tQ, Q );
        bidiag::ApplyP( RIGHT, NORMAL, A, tP, P );
        if( print )
        {
            Print( Q, "Q" );
            Print( P, "P" );
        }
        if( display )
        {
            Display( Q, "Q" );
            Display( P, "P" );
        }
    }

    // Reverse the accumulated Householder transforms
    bidiag::ApplyQ( LEFT,  ADJOINT, A, tQ, AOrig );
    bidiag::ApplyP( RIGHT, NORMAL,  A, tP, AOrig );
    if( print )
        Print( AOrig, "Manual bidiagonal" );
    if( display )
        Display( AOrig, "Manual bidiagonal" );

    // Compare the appropriate portion of AOrig and B
    if( m >= n )
    {
        MakeTrapezoidal( UPPER, AOrig );
        MakeTrapezoidal( LOWER, AOrig, 1 );
    }
    else
    {
        MakeTrapezoidal( LOWER, AOrig ); 
        MakeTrapezoidal( UPPER, AOrig, -1 );
    }
    B -= AOrig;
    if( print )
        Print( B, "Error in rotated bidiagonal" );
    if( display )
        Display( B, "Error in rotated bidiagonal" );
    const Real infNormError = InfinityNorm( B );
    const Real relError = infNormError / (Max(m,n)*oneNormAOrig*eps);

    OutputFromRoot
    (g.Comm(),
     "||B - Q^H A P||_oo / (max(m,n) || A ||_1 eps) = ",relError);
    PopIndent();

    // TODO: Use a more refined failure condition
    if( relError > Real(1) )
        LogicError("Relative error was unacceptably large");
}

template<typename F>
void TestBidiag
( const Grid& g,
  Int m,
  Int n,
  bool testCorrectness,
  bool print,
  bool display )
{
    OutputFromRoot(g.Comm(),"Testing with ",TypeName<F>());
    PushIndent();
    DistMatrix<F> A(g), AOrig(g);
    DistMatrix<F,STAR,STAR> tP(g), tQ(g);

    Uniform( A, m, n );
    if( testCorrectness )
        AOrig = A;
    if( print )
        Print( A, "A" );
    if( display )
        Display( A, "A" );

    OutputFromRoot(g.Comm(),"Starting bidiagonalization");
    mpi::Barrier( g.Comm() );
    Timer timer;
    timer.Start();
    Bidiag( A, tP, tQ );
    mpi::Barrier( g.Comm() );
    // TODO: Flop calculation
    OutputFromRoot(g.Comm(),"Time = ",timer.Stop()," seconds.");
    if( print )
    {
        Print( A, "A after Bidiag" );
        Print( tP, "tP after Bidiag" );
        Print( tQ, "tQ after Bidiag" );
    }
    if( display )
    {
        Display( A, "A after Bidiag" );
        Display( tP, "tP after Bidiag" );
        Display( tQ, "tQ after Bidiag" );
    }
    if( testCorrectness )
        TestCorrectness( A, tP, tQ, AOrig, print, display );
    PopIndent();
}

int 
main( int argc, char* argv[] )
{
    Environment env( argc, argv );
    mpi::Comm comm = mpi::COMM_WORLD;

    try
    {
        Int gridHeight = Input("--gridHeight","height of process grid",0);
        const bool colMajor = Input("--colMajor","column-major ordering?",true);
        const Int m = Input("--height","height of matrix",100);
        const Int n = Input("--width","width of matrix",100);
        const Int nb = Input("--nb","algorithmic blocksize",96);
        const bool testCorrectness = Input
            ("--correctness","test correctness?",true);
        const bool print = Input("--print","print matrices?",false);
        const bool display = Input("--display","display matrices?",false);
#ifdef EL_HAVE_MPC
        const mpfr_prec_t prec = Input("--prec","MPFR precision",256);
#endif
        ProcessInput();
        PrintInputReport();

#ifdef EL_HAVE_MPC
        mpc::SetPrecision( prec );
#endif

        if( gridHeight == 0 )
            gridHeight = Grid::FindFactor( mpi::Size(comm) );
        const GridOrder order = ( colMajor ? COLUMN_MAJOR : ROW_MAJOR );
        const Grid g( comm, gridHeight, order );
        SetBlocksize( nb );
        ComplainIfDebug();

        TestBidiag<float>( g, m, n, testCorrectness, print, display );
        TestBidiag<Complex<float>>( g, m, n, testCorrectness, print, display );

        TestBidiag<double>( g, m, n, testCorrectness, print, display );
        TestBidiag<Complex<double>>( g, m, n, testCorrectness, print, display );

#ifdef EL_HAVE_QD
        TestBidiag<DoubleDouble>( g, m, n, testCorrectness, print, display );
        TestBidiag<QuadDouble>( g, m, n, testCorrectness, print, display );
#endif

#ifdef EL_HAVE_QUAD
        TestBidiag<Quad>( g, m, n, testCorrectness, print, display );
        TestBidiag<Complex<Quad>>( g, m, n, testCorrectness, print, display );
#endif

#ifdef EL_HAVE_MPC
        TestBidiag<BigFloat>( g, m, n, testCorrectness, print, display );
#endif
    }
    catch( exception& e ) { ReportException(e); }

    return 0;
}
