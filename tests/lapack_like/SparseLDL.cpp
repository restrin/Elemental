/*
   Copyright (c) 2009-2012, Jack Poulson, Lexing Ying, and 
   The University of Texas at Austin.
   All rights reserved.

   Copyright (c) 2013, Jack Poulson, Lexing Ying, and Stanford University.
   All rights reserved.

   Copyright (c) 2013-2014, Jack Poulson and 
   The Georgia Institute of Technology.
   All rights reserved.

   Copyright (c) 2014-2015, Jack Poulson and Stanford University.
   All rights reserved.
   
   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>
using namespace El;

// TODO: Extend this driver to test with several different datatypes

int main( int argc, char* argv[] )
{
    Environment env( argc, argv );
    mpi::Comm comm = mpi::COMM_WORLD;
    const int commRank = mpi::Rank( comm );

    try
    {
        const Int n1 = Input("--n1","first grid dimension",30);
        const Int n2 = Input("--n2","second grid dimension",30);
        const Int n3 = Input("--n3","third grid dimension",30);
        const Int numRHS = Input("--numRHS","number of right-hand sides",5);
        const bool solve2d = Input("--solve2d","use 2d solve?",false);
        const bool selInv = Input("--selInv","selectively invert?",false);
        const bool intraPiv = Input("--intraPiv","pivot within fronts?",false);
        const bool natural = Input("--natural","analytical nested-diss?",true);
        const bool sequential = Input
            ("--sequential","sequential partitions?",true);
        const int numDistSeps = Input
            ("--numDistSeps",
             "number of separators to try per distributed partition",1);
        const int numSeqSeps = Input
            ("--numSeqSeps",
             "number of separators to try per sequential partition",1);
        const Int nbFact = Input("--nbFact","factorization blocksize",96);
        const Int nbSolve = Input("--nbSolve","solve blocksize",96);
        const Int cutoff = Input("--cutoff","cutoff for nested dissection",128);
        const bool unpack = Input("--unpack","unpack frontal matrix?",true);
        const bool print = Input("--print","print matrix?",false);
        const bool display = Input("--display","display matrix?",false);
        ProcessInput();

        BisectCtrl ctrl;
        ctrl.sequential = sequential;
        ctrl.numSeqSeps = numSeqSeps;
        ctrl.numDistSeps = numDistSeps;
        ctrl.cutoff = cutoff;

        const int N = n1*n2*n3;
        DistSparseMatrix<double> A(comm);
        Laplacian( A, n1, n2, n3 );
        A *= -1;
        if( display )
        {
            Display( A );
            Display( A.DistGraph() );
        }
        if( print )
        {
            Print( A );
            Print( A.DistGraph() );
        }

        Timer timer;

        OutputFromRoot(comm,"Generating random X and forming Y := A X...");
        timer.Start();
        DistMultiVec<double> X( N, numRHS, comm ), Y( N, numRHS, comm );
        MakeUniform( X );
        Zero( Y );
        Multiply( NORMAL, 1., A, X, 0., Y );
        Matrix<double> YOrigNorms;
        ColumnTwoNorms( Y, YOrigNorms );
        mpi::Barrier( comm );
        OutputFromRoot(comm,timer.Stop()," seconds");
        if( display )
        {
            Display( X, "X" );
            Display( Y, "Y" );
        }
        if( print )
        {
            Print( X, "X" );
            Print( Y, "Y" );
        }

        OutputFromRoot(comm,"Running nested dissection...");
        timer.Start();
        const auto& graph = A.DistGraph();
        ldl::DistNodeInfo info;
        ldl::DistSeparator sep;
        DistMap map, invMap;
        if( natural )
            ldl::NaturalNestedDissection
            ( n1, n2, n3, graph, map, sep, info, cutoff );
        else
            ldl::NestedDissection( graph, map, sep, info, ctrl );
        InvertMap( map, invMap );
        mpi::Barrier( comm );
        OutputFromRoot(comm,timer.Stop()," seconds");

        const Int rootSepSize = info.size;
        OutputFromRoot(comm,rootSepSize," vertices in root separator\n");
        /*
        if( display )
        {
            ostringstream osBefore, osAfter;
            osBefore << "Structure before fact. on process " << commRank;
            DisplayLocal( info, false, osBefore.str() );
            osAfter << "Structure after fact. on process " << commRank;
            DisplayLocal( info, true, osAfter.str() );
        }
        */

        OutputFromRoot(comm,"Building ldl::DistFront tree...");
        mpi::Barrier( comm );
        ldl::DistFront<double> front( A, map, sep, info );
        mpi::Barrier( comm );
        OutputFromRoot(comm,timer.Stop()," seconds");

        // Unpack the ldl::DistFront into a sparse matrix
        if( unpack )
        {
            DistSparseMatrix<double> APerm;
            front.Unpack( APerm, sep, info );
            MakeSymmetric( LOWER, APerm );
            if( print )
                Print( APerm, "APerm" );
            if( display )
                Display( APerm, "APerm" );
        }

        // Memory usage before factorization
        const Int localEntriesBefore = front.NumLocalEntries();
        const Int minLocalEntriesBefore = 
          mpi::AllReduce( localEntriesBefore, mpi::MIN, comm );
        const Int maxLocalEntriesBefore =
          mpi::AllReduce( localEntriesBefore, mpi::MAX, comm );
        const Int entriesBefore =
          mpi::AllReduce( localEntriesBefore, mpi::SUM, comm );
        OutputFromRoot
        (comm,
         "Memory usage before factorization: \n",Indent(),
         "  min entries:   ",minLocalEntriesBefore,"\n",Indent(),
         "  max entries:   ",maxLocalEntriesBefore,"\n",Indent(),
         "  total entries: ",entriesBefore,"\n");

        OutputFromRoot(comm,"Running LDL^T and redistribution...");
        SetBlocksize( nbFact );
        mpi::Barrier( comm );
        timer.Start();
        LDLFrontType type;
        if( solve2d )
        {
            if( intraPiv )
                type = ( selInv ? LDL_INTRAPIV_SELINV_2D : LDL_INTRAPIV_2D );
            else
                type = ( selInv ? LDL_SELINV_2D : LDL_2D );
        }
        else
        {
            if( intraPiv )
                type = ( selInv ? LDL_INTRAPIV_SELINV_1D : LDL_INTRAPIV_1D );
            else
                type = ( selInv ? LDL_SELINV_1D : LDL_1D );
        }
        LDL( info, front, type );
        mpi::Barrier( comm );
        const double factTime = timer.Stop();
        const double localFactGFlops = front.LocalFactorGFlops( selInv );
        const double factGFlops = mpi::AllReduce( localFactGFlops, comm ); 
        const double factSpeed = factGFlops / factTime;
        OutputFromRoot(comm,factTime," seconds, ",factSpeed," GFlop/s");

        // Memory usage after factorization
        const Int localEntriesAfter = front.NumLocalEntries();
        const Int minLocalEntriesAfter = 
          mpi::AllReduce( localEntriesAfter, mpi::MIN, comm );
        const Int maxLocalEntriesAfter =
          mpi::AllReduce( localEntriesAfter, mpi::MAX, comm );
        const Int entriesAfter =
          mpi::AllReduce( localEntriesAfter, mpi::SUM, comm );
        OutputFromRoot
        (comm,
         "Memory usage after factorization: \n",Indent(),
         "  min entries:   ",minLocalEntriesAfter,"\n",Indent(),
         "  max entries:   ",maxLocalEntriesAfter,"\n",Indent(),
         "  total entries: ",entriesAfter,"\n");

        OutputFromRoot(comm,"Solving against Y...");
        SetBlocksize( nbSolve );
        mpi::Barrier( comm );
        timer.Start();
        ldl::SolveAfter( invMap, info, front, Y );
        mpi::Barrier( comm );
        const double solveTime = timer.Stop();
        const double localSolveGFlops = front.LocalSolveGFlops( numRHS );
        const double solveGFlops = mpi::AllReduce( localSolveGFlops, comm ); 
        const double solveSpeed = solveGFlops / factTime;
        OutputFromRoot(comm,solveTime," seconds (",solveSpeed," GFlop/s)");

        OutputFromRoot(comm,"Checking error in computed solution...");
        Matrix<double> XNorms, YNorms;
        ColumnTwoNorms( X, XNorms );
        ColumnTwoNorms( Y, YNorms );
        Y -= X;
        Matrix<double> errorNorms;
        ColumnTwoNorms( Y, errorNorms );
        for( int j=0; j<numRHS; ++j )
            OutputFromRoot
            (comm,
             "Right-hand side ",j,":\n",Indent(),
             "|| x     ||_2 = ",XNorms.Get(j,0),"\n",Indent(),
             "|| error ||_2 = ",errorNorms.Get(j,0),"\n",Indent(),
             "|| A x   ||_2 = ",YOrigNorms.Get(j,0),"\n");
    }
    catch( exception& e ) { ReportException(e); }

    return 0;
}
