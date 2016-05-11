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
( bool print,
  const DistMatrix<F>& A,
  const DistMatrix<F,MD,STAR>& t,
  const DistMatrix<Base<F>,MD,STAR>& d,
        DistMatrix<F>& AOrig )
{
    typedef Base<F> Real;
    const Grid& g = A.Grid();
    const Int m = A.Height();
    const Int n = A.Width();
    const Int minDim = std::min(m,n);

    OutputFromRoot(g.Comm(),"Testing orthogonality of Q...");
    PushIndent();

    // Form Z := Q Q^H as an approximation to identity
    DistMatrix<F> Z(g);
    Identity( Z, m, n );
    lq::ApplyQ( RIGHT, NORMAL, A, t, d, Z );
    lq::ApplyQ( RIGHT, ADJOINT, A, t, d, Z );
    auto ZUpper = View( Z, 0, 0, minDim, minDim );

    // Form X := I - Q Q^H
    DistMatrix<F> X(g);
    Identity( X, minDim, minDim );
    X -= ZUpper;

    Real oneNormError = OneNorm( X );
    Real infNormError = InfinityNorm( X );
    Real frobNormError = FrobeniusNorm( X );
    OutputFromRoot
    (g.Comm(),
     "||Q Q^H - I||_1  = ",oneNormError,"\n",Indent(),
     "||Q Q^H - I||_oo = ",infNormError,"\n",Indent(),
     "||Q Q^H - I||_F  = ",frobNormError);
    PopIndent();

    OutputFromRoot(g.Comm(),"Testing if A = LQ...");
    PushIndent();

    // Form L Q
    auto L( A );
    MakeTrapezoidal( LOWER, L );
    lq::ApplyQ( RIGHT, NORMAL, A, t, d, L );

    // Form L Q - A
    L -= AOrig;
    
    const Real oneNormA = OneNorm( AOrig );
    const Real infNormA = InfinityNorm( AOrig );
    const Real frobNormA = FrobeniusNorm( AOrig );
    oneNormError = OneNorm( L );
    infNormError = InfinityNorm( L );
    frobNormError = FrobeniusNorm( L );
    OutputFromRoot
    (g.Comm(),
     "||A||_1       = ",oneNormA,"\n",Indent(),
     "||A||_oo      = ",infNormA,"\n",Indent(),
     "||A||_F       = ",frobNormA,"\n",Indent(),
     "||A - LQ||_1  = ",oneNormError,"\n",Indent(),
     "||A - LQ||_oo = ",infNormError,"\n",Indent(),
     "||A - LQ||_F  = ",frobNormError);
    PopIndent();
}

template<typename F>
void TestLQ( const Grid& g, Int m, Int n, bool testCorrectness, bool print )
{
    OutputFromRoot(g.Comm(),"Testing with ",TypeName<F>());
    PushIndent();
    DistMatrix<F> A(g), AOrig(g);
    Uniform( A, m, n );

    if( testCorrectness )
        AOrig = A;
    if( print )
        Print( A, "A" );
    DistMatrix<F,MD,STAR> t(g);
    DistMatrix<Base<F>,MD,STAR> d(g);

    OutputFromRoot(g.Comm(),"Starting LQ factorization...");
    mpi::Barrier( g.Comm() );
    Timer timer;
    timer.Start();
    LQ( A, t, d );
    mpi::Barrier( g.Comm() );
    const double runTime = timer.Stop();
    const double mD = double(m);
    const double nD = double(n);
    const double realGFlops = (2.*mD*mD*nD - 2./3.*mD*mD*mD)/(1.e9*runTime);
    const double gFlops = ( IsComplex<F>::value ? 4*realGFlops : realGFlops );
    OutputFromRoot(g.Comm(),runTime," seconds (",gFlops," GFlop/s)");
    if( print )
    {
        Print( A, "A after factorization" );
        Print( t, "phases" );
        Print( d, "diagonal" );
    }
    if( testCorrectness )
        TestCorrectness( print, A, t, d, AOrig );
    PopIndent();
}

int 
main( int argc, char* argv[] )
{
    Environment env( argc, argv );
    mpi::Comm comm = mpi::COMM_WORLD;

    try
    {
        int gridHeight = Input("--gridHeight","height of process grid",0);
        const bool colMajor = Input("--colMajor","column-major ordering?",true);
        const Int m = Input("--height","height of matrix",100);
        const Int n = Input("--width","width of matrix",100);
        const Int nb = Input("--nb","algorithmic blocksize",96);
        const bool testCorrectness = Input
            ("--correctness","test correctness?",true);
        const bool print = Input("--print","print matrices?",false);
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

        TestLQ<float>( g, m, n, testCorrectness, print );
        TestLQ<Complex<float>>( g, m, n, testCorrectness, print );

        TestLQ<double>( g, m, n, testCorrectness, print );
        TestLQ<Complex<double>>( g, m, n, testCorrectness, print );

#ifdef EL_HAVE_QD
        TestLQ<DoubleDouble>( g, m, n, testCorrectness, print );
        TestLQ<QuadDouble>( g, m, n, testCorrectness, print );
#endif

#ifdef EL_HAVE_QUAD
        TestLQ<Quad>( g, m, n, testCorrectness, print );
        TestLQ<Complex<Quad>>( g, m, n, testCorrectness, print );
#endif

#ifdef EL_HAVE_MPC
        TestLQ<BigFloat>( g, m, n, testCorrectness, print );
#endif
    }
    catch( exception& e ) { ReportException(e); }

    return 0;
}
