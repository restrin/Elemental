/*
   Copyright (c) 2009-2014, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
// NOTE: It is possible to simply include "elemental.hpp" instead
#include "elemental-lite.hpp"
#include ELEM_ENTRYWISEMAP_INC
#include ELEM_FROBENIUSNORM_INC
#include ELEM_PSEUDOSPECTRUM_INC
#include ELEM_GRCAR_INC
#include ELEM_FOXLI_INC
#include ELEM_HELMHOLTZPML_INC
#include ELEM_LOTKIN_INC
#include ELEM_UNIFORM_INC
using namespace std;
using namespace elem;

typedef double Real;
typedef Complex<Real> C;

int
main( int argc, char* argv[] )
{
    Initialize( argc, argv );

    try 
    {
        Int r = Input("--gridHeight","process grid height",0);
        const bool colMajor = Input("--colMajor","column-major ordering?",true);
        const Int matType = 
            Input("--matType","0:uniform,1:Haar,2:Lotkin,3:Grcar,4:FoxLi,"
                              "5:HelmholtzPML1D,6:HelmholtzPML2D",4);
        const Int n = Input("--size","height of matrix",100);
        const Int nbAlg = Input("--nbAlg","algorithmic blocksize",96);
#ifdef ELEM_HAVE_SCALAPACK
        const Int nbDist = Input("--nbDist","distribution blocksize",32);
#endif
        const Real realCenter = Input("--realCenter","real center",0.);
        const Real imagCenter = Input("--imagCenter","imag center",0.);
        const Real realWidth = Input("--realWidth","x width of image",0.);
        const Real imagWidth = Input("--imagWidth","y width of image",0.);
        const Int realSize = Input("--realSize","number of x samples",100);
        const Int imagSize = Input("--imagSize","number of y samples",100);
        const bool schur = Input("--schur","Schur decomposition?",true);
        const bool forceComplexSchur = 
            Input("--forceComplexSchur",
                  "switch to complex arithmetic for QR alg.",false);
        const bool forceComplexPs = 
            Input("--forceComplexPs",
                  "switch to complex arithmetic for PS iter's",true);
        const bool arnoldi = Input("--arnoldi","use Arnoldi?",true);
        const Int basisSize = Input("--basisSize","num Arnoldi vectors",10);
        const Int maxIts = Input("--maxIts","maximum pseudospec iter's",200);
        const Real tol = Input("--tol","tolerance for norm estimates",1e-6);
        const Real uniformRealCenter = 
            Input("--uniformRealCenter","real center of uniform dist",0.);
        const Real uniformImagCenter =
            Input("--uniformImagCenter","imag center of uniform dist",0.);
        const Real uniformRadius = 
            Input("--uniformRadius","radius of uniform dist",1.);
        const Int numBands = Input("--numBands","num bands for Grcar",3);
        const Real omega = Input("--omega","frequency for Fox-Li/Helm",16*M_PI);
        const Int mx = Input("--mx","number of x points for HelmholtzPML",30);
        const Int my = Input("--my","number of y points for HelmholtzPML",30);
        const Int numPmlPoints = Input("--numPml","num PML points for Helm",5);
        const double sigma = Input("--sigma","PML amplitude",1.5);
        const double pmlExp = Input("--pmlExp","PML takeoff exponent",3.);
        const bool progress = Input("--progress","print progress?",true);
        const bool deflate = Input("--deflate","deflate?",true);
        const bool print = Input("--print","print matrices?",false);
        const bool display = Input("--display","display matrices?",false);
        const bool write = Input("--write","write matrices?",false);
        const bool writePseudo = Input("--writePs","write pseudospec.",false);
        const Int numFreq = Input("--numFreq","numerical save frequency",0);
        const Int imgFreq = Input("--imgFreq","image save frequency",0);
        const std::string numBase =
            Input("--numBase","numerical save basename",std::string("snap"));
        const std::string imgBase =
            Input("--imgBase","image save basename",std::string("logSnap"));
        const Int numFormatInt = Input("--numFormat","numerical format",2);
        const Int imgFormatInt = Input("--imgFormat","image format",8);
        const Int colorMapInt = Input("--colorMap","color map",0);
        ProcessInput();
        PrintInputReport();

        if( r == 0 )
            r = Grid::FindFactor( mpi::Size(mpi::COMM_WORLD) );
        const GridOrder order = ( colMajor ? COLUMN_MAJOR : ROW_MAJOR );
        const Grid g( mpi::COMM_WORLD, r, order );
        SetBlocksize( nbAlg );
#ifdef ELEM_HAVE_SCALAPACK
        SetDefaultBlockHeight( nbDist );
        SetDefaultBlockWidth( nbDist );
#endif
        if( numFormatInt < 1 || numFormatInt >= FileFormat_MAX )
            LogicError("Invalid numerical format integer, should be in [1,",
                       FileFormat_MAX,")");
        if( imgFormatInt < 1 || imgFormatInt >= FileFormat_MAX )
            LogicError("Invalid image format integer, should be in [1,",
                       FileFormat_MAX,")");

        const FileFormat numFormat = static_cast<FileFormat>(numFormatInt);
        const FileFormat imgFormat = static_cast<FileFormat>(imgFormatInt);
        const ColorMap colorMap = static_cast<ColorMap>(colorMapInt);
        SetColorMap( colorMap );
        const C center(realCenter,imagCenter);
        const C uniformCenter(uniformRealCenter,uniformImagCenter);

        bool isReal = true;
        std::string matName;
        DistMatrix<Real> AReal(g);
        DistMatrix<C> ACpx(g);
        switch( matType )
        {
        case 0: matName="uniform";
                Uniform( ACpx, n, n, uniformCenter, uniformRadius );
                isReal = false;
                break;
        case 1: matName="Haar";
                Haar( ACpx, n );
                isReal = false;
                break;
        case 2: matName="Lotkin";
                Lotkin( AReal, n );
                isReal = true;
                break;
        case 3: matName="Grcar";
                Grcar( AReal, n, numBands );
                isReal = true;
                break;
        case 4: matName="FoxLi";
                FoxLi( ACpx, n, omega );
                isReal = false;
                break;
        case 5: matName="HelmholtzPML";
                HelmholtzPML
                ( ACpx, n, C(omega), numPmlPoints, sigma, pmlExp );
                isReal = false;
                break;
        case 6: matName="HelmholtzPML2D";
                HelmholtzPML
                ( ACpx, mx, my, C(omega), numPmlPoints, sigma, pmlExp );
                isReal = false;
                break;
        default: LogicError("Invalid matrix type");
        }
        if( display )
        {
            if( isReal )
                Display( AReal, "A" );
            else
                Display( ACpx, "A" );
        }
        if( write )
        {
            if( isReal )
            {
                Write( AReal, "A", numFormat );
                Write( AReal, "A", imgFormat );
            }
            else
            {
                Write( ACpx, "A", numFormat );
                Write( ACpx, "A", imgFormat );
            }
        }

        PseudospecCtrl<Real> psCtrl;
        psCtrl.schur = schur;
        psCtrl.forceComplexSchur = forceComplexSchur;
        psCtrl.forceComplexPs = forceComplexPs;
        psCtrl.maxIts = maxIts;
        psCtrl.tol = tol;
        psCtrl.deflate = deflate;
        psCtrl.arnoldi = arnoldi;
        psCtrl.basisSize = basisSize;
        psCtrl.progress = progress;
        psCtrl.snapCtrl.imgFreq = imgFreq;
        psCtrl.snapCtrl.numFreq = numFreq;
        psCtrl.snapCtrl.imgFormat = imgFormat;
        psCtrl.snapCtrl.numFormat = numFormat;
        psCtrl.snapCtrl.imgBase = imgBase;
        psCtrl.snapCtrl.numBase = numBase;

        // Visualize the pseudospectrum by evaluating ||inv(A-sigma I)||_2 
        // for a grid of complex sigma's.
        DistMatrix<Real> invNormMap(g);
        DistMatrix<Int> itCountMap(g);
        if( realWidth != 0. && imagWidth != 0. )
        {
            if( isReal )
                itCountMap = Pseudospectrum
                ( AReal, invNormMap, center, realWidth, imagWidth, 
                  realSize, imagSize, psCtrl );
            else
                itCountMap = Pseudospectrum
                ( ACpx, invNormMap, center, realWidth, imagWidth, 
                  realSize, imagSize, psCtrl );
        }
        else
        {
            if( isReal )
                itCountMap = Pseudospectrum
                ( AReal, invNormMap, center, realSize, imagSize, psCtrl );
            else
                itCountMap = Pseudospectrum
                ( ACpx, invNormMap, center, realSize, imagSize, psCtrl );
        }
        const Int numIts = MaxNorm( itCountMap );
        if( mpi::WorldRank() == 0 )
            std::cout << "num iterations=" << numIts << std::endl;
        if( display )
        {
            Display( invNormMap, "invNormMap" );
            Display( itCountMap, "itCountMap" );
        }
        if( write || writePseudo )
        {
            Write( invNormMap, "invNormMap", numFormat );
            Write( invNormMap, "invNormMap", imgFormat );
            Write( itCountMap, "itCountMap", numFormat );
            Write( itCountMap, "itCountMap", imgFormat );
        }

        // Take the entrywise log
        EntrywiseMap( invNormMap, []( Real alpha ) { return Log(alpha); } );
        if( display )
        {
            Display( invNormMap, "logInvNormMap" );
            if( GetColorMap() != GRAYSCALE_DISCRETE )
            {
                auto colorMap = GetColorMap();
                SetColorMap( GRAYSCALE_DISCRETE );
                Display( invNormMap, "discreteLogInvNormMap" );
                SetColorMap( colorMap );
            }
        }
        if( write || writePseudo )
        {
            Write( invNormMap, "logInvNormMap", numFormat );
            Write( invNormMap, "logInvNormMap", imgFormat );
            if( GetColorMap() != GRAYSCALE_DISCRETE )
            {
                auto colorMap = GetColorMap();
                SetColorMap( GRAYSCALE_DISCRETE );
                Write( invNormMap, "discreteLogInvNormMap", numFormat ); 
                Write( invNormMap, "discreteLogInvNormMap", imgFormat ); 
                SetColorMap( colorMap );
            }
        }
    }
    catch( exception& e ) { ReportException(e); }

    Finalize();
    return 0;
}
