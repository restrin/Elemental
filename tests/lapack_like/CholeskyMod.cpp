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
(       UpperOrLower uplo,
  const Matrix<F>& T,
        Base<F> alpha,
  const Matrix<F>& V,
  const Matrix<F>& A,
        Int numRHS=100 )
{
    typedef Base<F> Real;
    const Int n = V.Height();
    const Real eps = limits::Epsilon<Real>();

    Matrix<F> B( A );
    Herk( uplo, NORMAL, alpha, V, Real(1), B );

    // Test correctness by multiplying a random set of vectors by 
    // A + alpha V V^H, then using the Cholesky factorization to solve.
    Matrix<F> X, Y;
    Uniform( X, n, numRHS );
    Zeros( Y, n, numRHS );
    Hemm( LEFT, uplo, F(1), B, X, F(0), Y );
    const Real oneNormY = OneNorm( Y );

    cholesky::SolveAfter( uplo, NORMAL, T, Y );
    X -= Y;
    const Real infNormE = InfinityNorm( X );
    const Real relError = infNormE / (eps*n*oneNormY);

    Output("|| X - B \\ Y ||_oo / (n eps || Y ||_1) = ",relError);

    // TODO: Use a more refined failure condition
    if( relError > Real(1) )
        LogicError("Relative error was unacceptably large");
}

template<typename F>
void TestCorrectness
(       UpperOrLower uplo,
  const DistMatrix<F>& T,
        Base<F> alpha,
  const DistMatrix<F>& V,
  const DistMatrix<F>& A,
        Int numRHS=100 )
{
    typedef Base<F> Real;
    const Int n = V.Height();
    const Real eps = limits::Epsilon<Real>();
    const Grid& g = T.Grid();

    DistMatrix<F> B( A );
    Herk( uplo, NORMAL, alpha, V, Real(1), B );

    // Test correctness by multiplying a random set of vectors by 
    // A + alpha V V^H, then using the Cholesky factorization to solve.
    DistMatrix<F> X(g), Y(g);
    Uniform( X, n, numRHS );
    Zeros( Y, n, numRHS );
    Hemm( LEFT, uplo, F(1), B, X, F(0), Y );
    const Real oneNormY = OneNorm( Y );

    cholesky::SolveAfter( uplo, NORMAL, T, Y );
    X -= Y;
    const Real infNormE = InfinityNorm( X );
    const Real relError = infNormE / (eps*n*oneNormY);

    OutputFromRoot
    (g.Comm(),"|| X - B \\ Y ||_oo / (n eps || Y ||_1) = ",relError);

    // TODO: Use a more refined failure condition
    if( relError > Real(1) )
        LogicError("Relative error was unacceptably large");
}

template<typename F> 
void TestCholeskyMod
( UpperOrLower uplo,
  Int m,
  Int n, 
  Base<F> alpha,
  bool correctness,
  bool print )
{
    Output("Testing with ",TypeName<F>());
    PushIndent();

    Matrix<F> T, A;
    HermitianUniformSpectrum( T, m, 1e-9, 10 );
    if( correctness )
        A = T;
    if( print )
        Print( T, "A" );

    Output("Starting Cholesky...");
    Timer timer;
    timer.Start();
    Cholesky( uplo, T );
    double runTime = timer.Stop();
    double realGFlops = 1./3.*Pow(double(m),3.)/(1.e9*runTime);
    double gFlops = ( IsComplex<F>::value ? 4*realGFlops : realGFlops );
    Output(runTime," seconds (",gFlops," GFlop/s)");
    MakeTrapezoidal( uplo, T );
    if( print )
        Print( T, "Cholesky factor" );

    Matrix<F> V, VMod;
    Uniform( V, m, n );
    V *= F(1)/Sqrt(F(m)*F(n));
    VMod = V;
    if( print )
        Print( V, "V" );

    Output("Starting Cholesky mod...");
    timer.Start();
    CholeskyMod( uplo, T, alpha, VMod );
    runTime = timer.Stop();
    Output(runTime," seconds");
    if( print )
        Print( T, "Modified Cholesky factor" );

    if( correctness )
        TestCorrectness( uplo, T, alpha, V, A );
    PopIndent();
}

template<typename F> 
void TestCholeskyMod
( const Grid& g,
  UpperOrLower uplo,
  Int m,
  Int n, 
  Base<F> alpha,
  bool correctness,
  bool print )
{
    OutputFromRoot(g.Comm(),"Testing with ",TypeName<F>());
    PushIndent();

    DistMatrix<F> T(g), A(g);
    HermitianUniformSpectrum( T, m, 1e-9, 10 );
    if( correctness )
        A = T;
    if( print )
        Print( T, "A" );

    OutputFromRoot(g.Comm(),"Starting Cholesky...");
    Timer timer;
    timer.Start();
    Cholesky( uplo, T );
    double runTime = timer.Stop();
    double realGFlops = 1./3.*Pow(double(m),3.)/(1.e9*runTime);
    double gFlops = ( IsComplex<F>::value ? 4*realGFlops : realGFlops );
    OutputFromRoot(g.Comm(),runTime," seconds (",gFlops," GFlop/s)");
    MakeTrapezoidal( uplo, T );
    if( print )
        Print( T, "Cholesky factor" );

    DistMatrix<F> V(g), VMod(g);
    Uniform( V, m, n );
    V *= F(1)/Sqrt(F(m)*F(n));
    VMod = V;
    if( print )
        Print( V, "V" );

    OutputFromRoot(g.Comm(),"Starting Cholesky mod...");
    timer.Start();
    CholeskyMod( uplo, T, alpha, VMod );
    runTime = timer.Stop();
    OutputFromRoot(g.Comm(),runTime," seconds");
    if( print )
        Print( T, "Modified Cholesky factor" );

    if( correctness )
        TestCorrectness( uplo, T, alpha, V, A );
    PopIndent();
}

int 
main( int argc, char* argv[] )
{
    Environment env( argc, argv );
    mpi::Comm comm = mpi::COMM_WORLD;

    try
    {
        Int gridHeight = Input("--gridHeight","process grid height",0);
        const bool colMajor = Input("--colMajor","column-major ordering?",true);
        const char uploChar = Input("--uplo","upper or lower storage: L/U",'L');
        const Int m = Input("--m","height of matrix",100);
        const Int n = Input("--n","rank of update",5);
        const Int nb = Input("--nb","algorithmic blocksize",96);
        const double alpha = Input("--alpha","update scaling",3.);
        const bool sequential = 
          Input("--sequential","test sequential?",true);
        const bool correctness =
          Input("--correctness","test correctness?",true);
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
        const UpperOrLower uplo = CharToUpperOrLower( uploChar );
        SetBlocksize( nb );
        ComplainIfDebug();

        if( sequential && mpi::Rank() == 0 )
        {
            TestCholeskyMod<float>
            ( uplo, m, n, alpha, correctness, print );
            TestCholeskyMod<Complex<float>>
            ( uplo, m, n, alpha, correctness, print );

            TestCholeskyMod<double>
            ( uplo, m, n, alpha, correctness, print );
            TestCholeskyMod<Complex<double>>
            ( uplo, m, n, alpha, correctness, print );

#ifdef EL_HAVE_QD
            TestCholeskyMod<DoubleDouble>
            ( uplo, m, n, alpha, correctness, print );
            TestCholeskyMod<QuadDouble>
            ( uplo, m, n, alpha, correctness, print );
#endif

#ifdef EL_HAVE_QUAD
            TestCholeskyMod<Quad>
            ( uplo, m, n, alpha, correctness, print );
            TestCholeskyMod<Complex<Quad>>
            ( uplo, m, n, alpha, correctness, print );
#endif

#ifdef EL_HAVE_MPC
            TestCholeskyMod<BigFloat>
            ( uplo, m, n, alpha, correctness, print );
#endif
        }

        TestCholeskyMod<float>
        ( g, uplo, m, n, alpha, correctness, print );
        TestCholeskyMod<Complex<float>>
        ( g, uplo, m, n, alpha, correctness, print );

        TestCholeskyMod<double>
        ( g, uplo, m, n, alpha, correctness, print );
        TestCholeskyMod<Complex<double>>
        ( g, uplo, m, n, alpha, correctness, print );

#ifdef EL_HAVE_QD
        TestCholeskyMod<DoubleDouble>
        ( g, uplo, m, n, alpha, correctness, print );
        TestCholeskyMod<QuadDouble>
        ( g, uplo, m, n, alpha, correctness, print );
#endif

#ifdef EL_HAVE_QUAD
        TestCholeskyMod<Quad>
        ( g, uplo, m, n, alpha, correctness, print );
        TestCholeskyMod<Complex<Quad>>
        ( g, uplo, m, n, alpha, correctness, print );
#endif

#ifdef EL_HAVE_MPC
        TestCholeskyMod<BigFloat>
        ( g, uplo, m, n, alpha, correctness, print );
#endif
    }
    catch( exception& e ) { ReportException(e); }

    return 0;
}
