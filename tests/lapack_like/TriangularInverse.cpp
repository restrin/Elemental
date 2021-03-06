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
  UpperOrLower uplo,
  UnitOrNonUnit diag,
  const DistMatrix<F>& A,
  const DistMatrix<F>& AOrig )
{
    typedef Base<F> Real;
    const Grid& g = A.Grid();
    const Int m = AOrig.Height();

    DistMatrix<F> X(g), Y(g);
    Uniform( X, m, 100 );
    Y = X;

    // Since A o A^-1 = I, test the change introduced by the approximate comp.
    Trmm( LEFT, uplo, NORMAL, diag, F(1), A,     Y );
    Trmm( LEFT, uplo, NORMAL, diag, F(1), AOrig, Y );
    Y -= X;

    const Real oneNormOrig = OneNorm( AOrig );
    const Real infNormOrig = InfinityNorm( AOrig );
    const Real frobNormOrig = FrobeniusNorm( AOrig );
    const Real oneNormFinal = OneNorm( A );
    const Real infNormFinal = InfinityNorm( A );
    const Real frobNormFinal = FrobeniusNorm( A );
    const Real oneNormError = OneNorm( Y );
    const Real infNormError = InfinityNorm( Y );
    const Real frobNormError = FrobeniusNorm( Y );
    OutputFromRoot
    (g.Comm(),
     "||A||_1           = ",oneNormOrig,"\n",Indent(),
     "||A||_oo          = ",infNormOrig,"\n",Indent(),
     "||A||_F           = ",frobNormOrig,"\n",Indent(),
     "||A^-1||_1        = ",oneNormFinal,"\n",Indent(),
     "||A^-1||_oo       = ",infNormFinal,"\n",Indent(),
     "||A^-1||_F        = ",frobNormFinal,"\n",Indent(),
     "||A A^-1 - I||_1  = ",oneNormError,"\n",Indent(),
     "||A A^-1 - I||_oo = ",infNormError,"\n",Indent(),
     "||A A^-1 - I||_F  = ",frobNormError);
}

template<typename F> 
void TestTriangularInverse
( const Grid& g,
  UpperOrLower uplo,
  UnitOrNonUnit diag,
  Int m,
  bool testCorrectness,
  bool print )
{
    OutputFromRoot(g.Comm(),"Testing with ",TypeName<F>());
    PushIndent();

    DistMatrix<F> A(g), AOrig(g);
    HermitianUniformSpectrum( A, m, 1, 10 );
    MakeTrapezoidal( uplo, A );
    if( testCorrectness )
        AOrig = A;
    if( print )
        Print( A, "A" );

    OutputFromRoot(g.Comm(),"Starting triangular inversion...");
    mpi::Barrier( g.Comm() );
    Timer timer;
    timer.Start();
    TriangularInverse( uplo, diag, A );
    mpi::Barrier( g.Comm() );
    const double runTime = timer.Stop();
    const double realGFlops = 1./3.*Pow(double(m),3.)/(1.e9*runTime);
    const double gFlops = ( IsComplex<F>::value ? 4*realGFlops : realGFlops );
    OutputFromRoot(g.Comm(),"Time = ",runTime," seconds (",gFlops," GFlop/s)");
    if( print )
        Print( A, "A after inversion" );
    if( testCorrectness )
        TestCorrectness( print, uplo, diag, A, AOrig );
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
        const char uploChar = Input("--uplo","upper or lower storage: L/U",'L');
        const char diagChar = Input("--diag","(non-)unit diagonal: N/U",'N');
        const Int m = Input("--height","height of matrix",100);
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

        const UpperOrLower uplo = CharToUpperOrLower( uploChar );
        const UnitOrNonUnit diag = CharToUnitOrNonUnit( diagChar );

        if( gridHeight == 0 )
            gridHeight = Grid::FindFactor( mpi::Size(comm) );
        const GridOrder order = ( colMajor ? COLUMN_MAJOR : ROW_MAJOR );
        const Grid g( comm, gridHeight, order );
        SetBlocksize( nb );
        ComplainIfDebug();
        OutputFromRoot
        (g.Comm(),"Will test TriangularInverse",uploChar,diagChar);

        TestTriangularInverse<float>
        ( g, uplo, diag, m, testCorrectness, print );
        TestTriangularInverse<Complex<float>>
        ( g, uplo, diag, m, testCorrectness, print );

        TestTriangularInverse<double>
        ( g, uplo, diag, m, testCorrectness, print );
        TestTriangularInverse<Complex<double>>
        ( g, uplo, diag, m, testCorrectness, print );

#ifdef EL_HAVE_QD
        TestTriangularInverse<DoubleDouble>
        ( g, uplo, diag, m, testCorrectness, print );
        TestTriangularInverse<QuadDouble>
        ( g, uplo, diag, m, testCorrectness, print );
#endif

#ifdef EL_HAVE_QUAD
        TestTriangularInverse<Quad>
        ( g, uplo, diag, m, testCorrectness, print );
        TestTriangularInverse<Complex<Quad>>
        ( g, uplo, diag, m, testCorrectness, print );
#endif

#ifdef EL_HAVE_MPC
        TestTriangularInverse<BigFloat>
        ( g, uplo, diag, m, testCorrectness, print );
#endif
    }
    catch( exception& e ) { ReportException(e); }

    return 0;
}
