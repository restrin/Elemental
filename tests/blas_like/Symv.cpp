/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>
using namespace El;

template<typename T>
void TestSymv
( UpperOrLower uplo,
  Int m,
  T alpha,
  T beta,
  bool print,
  const Grid& g,
  Int nbLocal )
{
    OutputFromRoot(g.Comm(),"Testing with ",TypeName<T>());
    PushIndent();

    SetLocalSymvBlocksize<T>( nbLocal );

    DistMatrix<T> A(g), x(g), y(g);

    Uniform( A, m, m );
    Uniform( x, m, 1 );
    Uniform( y, m, 1 );
    if( print )
    {
        Print( A, "A" );
        Print( x, "x" );
        Print( y, "y" );
    }

    // Test Symv
    OutputFromRoot(g.Comm(),"Starting Symv");
    mpi::Barrier( g.Comm() );
    Timer timer;
    timer.Start();
    Symv( uplo, alpha, A, x, beta, y );
    mpi::Barrier( g.Comm() );
    const double runTime = timer.Stop();
    const double realGFlops = 2.*double(m)*double(m)/(1.e9*runTime);
    const double gFlops = ( IsComplex<T>::value ? 4*realGFlops : realGFlops );
    OutputFromRoot
    (g.Comm(),"Finished in ",runTime," seconds (",gFlops," GFlop/s");
    if( print )
        Print( y, BuildString("y := ",alpha," Symm(A) x + ",beta," y") );

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
        const Int m = Input("--m","height of matrix",100);
        const Int nb = Input("--nb","algorithmic blocksize",96);
        const Int nbLocal = Input("--nbLocal","local blocksize",32);
        const bool print = Input("--print","print matrices?",false);
        ProcessInput();
        PrintInputReport();

        if( gridHeight == 0 )
            gridHeight = Grid::DefaultHeight( mpi::Size(comm) );
        const GridOrder order = ( colMajor ? COLUMN_MAJOR : ROW_MAJOR );
        const Grid g( comm, gridHeight, order );
        const UpperOrLower uplo = CharToUpperOrLower( uploChar );
        SetBlocksize( nb );

        ComplainIfDebug();
        OutputFromRoot(comm,"Will test Symv ",uploChar);

        TestSymv<float>
        ( uplo, m,
          float(3), float(4),
          print, g, nbLocal );
        TestSymv<Complex<float>>
        ( uplo, m,
          Complex<float>(3), Complex<float>(4),
          print, g, nbLocal );

        TestSymv<double>
        ( uplo, m,
          double(3), double(4),
          print, g, nbLocal );
        TestSymv<Complex<double>>
        ( uplo, m,
          Complex<double>(3), Complex<double>(4),
          print, g, nbLocal );

#ifdef EL_HAVE_QD
        TestSymv<DoubleDouble>
        ( uplo, m,
          DoubleDouble(3), DoubleDouble(4),
          print, g, nbLocal );
        TestSymv<QuadDouble>
        ( uplo, m,
          QuadDouble(3), QuadDouble(4),
          print, g, nbLocal );

        TestSymv<Complex<DoubleDouble>>
        ( uplo, m,
          Complex<DoubleDouble>(3), Complex<DoubleDouble>(4),
          print, g, nbLocal );
        TestSymv<Complex<QuadDouble>>
        ( uplo, m,
          Complex<QuadDouble>(3), Complex<QuadDouble>(4),
          print, g, nbLocal );
#endif

#ifdef EL_HAVE_QUAD
        TestSymv<Quad>
        ( uplo, m,
          Quad(3), Quad(4),
          print, g, nbLocal );
        TestSymv<Complex<Quad>>
        ( uplo, m,
          Complex<Quad>(3), Complex<Quad>(4),
          print, g, nbLocal );
#endif

#ifdef EL_HAVE_MPC
        TestSymv<BigFloat>
        ( uplo, m,
          BigFloat(3), BigFloat(4),
          print, g, nbLocal );
        TestSymv<Complex<BigFloat>>
        ( uplo, m,
          Complex<BigFloat>(3), Complex<BigFloat>(4),
          print, g, nbLocal );
#endif
    }
    catch( exception& e ) { ReportException(e); }

    return 0;
}
