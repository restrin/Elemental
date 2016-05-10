/*
   Copyright (c) 2009-2016, Jack Poulson, Lexing Ying,
   The University of Texas at Austin, Stanford University, and the
   Georgia Insitute of Technology.
   All rights reserved.
 
   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>
using namespace El;

// TODO: Extend this driver to test with several different datatypes

template<typename F>
void MakeFrontsUniform( ldl::Front<F>& front )
{
    ldl::ChangeFrontType( front, SYMM_2D );
    MakeUniform( front.LDense );
    for( ldl::Front<F>* child : front.children )
        MakeFrontsUniform( *child );
}

template<typename F>
void MakeFrontsUniform( ldl::DistFront<F>& front )
{
    ldl::ChangeFrontType( front, SYMM_2D );
    MakeUniform( front.L2D );
    if( front.child != nullptr )
        MakeFrontsUniform( *front.child );
    else
        MakeFrontsUniform( *front.duplicate );
}

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
        const Int numRepeats = Input
            ("--numRepeats","number of repeated factorizations",5);
        const bool intraPiv = Input("--intraPiv","frontal pivoting?",false);
        const bool sequential = Input
            ("--sequential","sequential partitions?",true);
        const int numDistSeps = Input
            ("--numDistSeps",
             "number of partitions to try per distributed partition",1);
        const int numSeqSeps = Input
            ("--numSeqSeps",
             "number of partitions to try per sequential partition",1);
        const Int cutoff = Input("--cutoff","cutoff for nested dissection",128);
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

        OutputFromRoot(comm,"Running nested dissection...");
        timer.Start();
        const DistGraph& graph = A.DistGraph();
        ldl::DistNodeInfo info;
        ldl::DistSeparator sep;
        DistMap map, invMap;
        ldl::NestedDissection( graph, map, sep, info, ctrl );
        InvertMap( map, invMap );
        mpi::Barrier( comm );
        OutputFromRoot(comm,timer.Stop()," seconds");

        const Int rootSepSize = info.size;
        OutputFromRoot(comm,rootSepSize," vertices in root separator\n");

        OutputFromRoot(comm,"Building ldl::DistFront tree...");
        mpi::Barrier( comm );
        timer.Start();
        ldl::DistFront<double> front( A, map, sep, info, false );
        mpi::Barrier( comm );
        OutputFromRoot(comm,timer.Stop()," seconds");

        for( Int repeat=0; repeat<numRepeats; ++repeat )
        {
            if( repeat != 0 )
                MakeFrontsUniform( front );

            OutputFromRoot(comm,"Running LDL^T and redistribution...");
            mpi::Barrier( comm );
            timer.Start();
            if( intraPiv )
                LDL( info, front, LDL_INTRAPIV_1D );
            else
                LDL( info, front, LDL_1D );
            mpi::Barrier( comm );
            OutputFromRoot(comm,timer.Stop()," seconds");

            OutputFromRoot(comm,"Solving against random right-hand side...");
            timer.Start();
            DistMultiVec<double> y( N, 1, comm );
            MakeUniform( y );
            ldl::SolveAfter( invMap, info, front, y );
            mpi::Barrier( comm );
            OutputFromRoot(comm,"Time = ",timer.Stop()," seconds");

            // TODO: Check residual error
        }
    }
    catch( exception& e ) { ReportException(e); }

    return 0;
}
