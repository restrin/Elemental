/*
   Copyright 2009-2011, Jack Poulson.
   All rights reserved.

   Copyright 2011-2012, Jack Poulson, Lexing Ying, and 
   The University of Texas at Austin.
   All rights reserved.

   Copyright 2013, Jack Poulson, Lexing Ying, and Stanford University.
   All rights reserved.

   Copyright 2013-2014, Jack Poulson and The Georgia Institute of Technology.
   All rights reserved.

   Copyright 2014-2015, Jack Poulson and Stanford University.
   All rights reserved.
   
   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

namespace El {
namespace ldl {

inline void PairwiseExchangeLowerStruct
( Int& theirSize, vector<Int>& theirLowerStruct, const DistNodeInfo& node )
{
    // Determine our partner's rank for this exchange in node's communicator
    const int teamRank = mpi::Rank( node.comm );
    const int teamSize = mpi::Size( node.comm );
    const int childTeamRank = mpi::Rank( node.child->comm );
    const int myTeamSize = mpi::Size( node.child->comm );
    const int otherTeamSize = teamSize - myTeamSize;
    const bool inFirstTeam = ( teamRank == childTeamRank );
    const int partner =
        ( inFirstTeam ? teamRank+myTeamSize : teamRank-otherTeamSize );

    // SendRecv the message lengths
    const Int mySize = node.child->size;
    const Int myLowerStructSize = node.child->lowerStruct.size();
    const Int initialSends[2] = { mySize, myLowerStructSize };
    Int initialRecvs[2];
    mpi::SendRecv
    ( initialSends, 2, partner,
      initialRecvs, 2, partner, node.comm );
    theirSize = initialRecvs[0];
    const Int theirLowerStructSize = initialRecvs[1];

    // Perform the exchange
    theirLowerStruct.resize( theirLowerStructSize );
    mpi::SendRecv
    ( &node.child->lowerStruct[0], myLowerStructSize, partner,
      &theirLowerStruct[0], theirLowerStructSize, partner, node.comm );
}

inline void BroadcastLowerStruct
( Int& theirSize, vector<Int>& theirLowerStruct, const DistNodeInfo& node )
{
    // Determine our partner's rank for this exchange in node's communicator
    const int teamRank = mpi::Rank( node.comm );
    const int teamSize = mpi::Size( node.comm );
    const int childTeamRank = mpi::Rank( node.child->comm );
    const int myTeamSize = mpi::Size( node.child->comm );
    const int otherTeamSize = teamSize - myTeamSize;
    const bool inFirstTeam = ( teamRank == childTeamRank );

    if( childTeamRank == 0 )
    {
        const int partner =
            ( inFirstTeam ? teamRank+myTeamSize : teamRank-otherTeamSize );

        // SendRecv the message lengths
        const Int mySize = node.child->size;
        const Int myLowerStructSize = node.child->lowerStruct.size();
        const Int initialSends[2] = { mySize, myLowerStructSize };
        Int initialRecvs[2];
        mpi::SendRecv
        ( initialSends, 2, partner, 
          initialRecvs, 2, partner, node.comm );
        theirSize = initialRecvs[0];
        const Int theirLowerStructSize = initialRecvs[1];

        // Perform the exchange
        theirLowerStruct.resize( theirLowerStructSize );
        mpi::SendRecv
        ( &node.child->lowerStruct[0], myLowerStructSize, partner,
          &theirLowerStruct[0], theirLowerStructSize, partner, node.comm );

        // Broadcast the other team's child's sizes
        mpi::Broadcast( initialRecvs, 2, 0, node.child->comm );

        // Broadcast the other team's child's lower struct
        mpi::Broadcast
        ( &theirLowerStruct[0], theirLowerStructSize, 0, node.child->comm );
    } 
    else
    {
        // Receive the other team's child's sizes
        Int initialRecvs[2];
        mpi::Broadcast( initialRecvs, 2, 0, node.child->comm );
        theirSize = initialRecvs[0];
        const Int theirLowerStructSize = initialRecvs[1];

        // Receive the other team's child's lower struct
        theirLowerStruct.resize( theirLowerStructSize );
        mpi::Broadcast
        ( &theirLowerStruct[0], theirLowerStructSize, 0, node.child->comm );
    }
}

inline void GetLowerStruct
( Int& theirSize, vector<Int>& theirLowerStruct, const DistNodeInfo& node )
{
    const int teamSize = mpi::Size( node.comm );
    const int childTeamSize = mpi::Size( node.child->comm );
    const int leftTeamSize =
        ( node.child->onLeft ? childTeamSize : teamSize-childTeamSize );
    const int rightTeamSize = teamSize - leftTeamSize;
    if( leftTeamSize == rightTeamSize )
        PairwiseExchangeLowerStruct( theirSize, theirLowerStruct, node );
    else
        BroadcastLowerStruct( theirSize, theirLowerStruct, node );
}

inline void ComputeStructAndRelInds
( Int theirSize, const vector<Int>& theirLowerStruct, DistNodeInfo& node )
{
    const auto& myLowerStruct = node.child->lowerStruct;
    DEBUG_ONLY(
      if( !IsStrictlySorted(myLowerStruct) )
      {
          if( IsSorted(myLowerStruct) )
              LogicError("Repeat in my lower struct");
          else
              LogicError("My lower struct not sorted");
      }
      if( !IsStrictlySorted(theirLowerStruct) )
      {
          if( IsSorted(theirLowerStruct) )
              LogicError("Repeat in their lower struct");
          else
              LogicError("Their lower struct not sorted");
      }
      if( !IsStrictlySorted(node.origLowerStruct) )
      {
          if( IsSorted(node.origLowerStruct) )
              LogicError("Repeat in original struct");
          else
              LogicError("Original struct not sorted");
      }
    )

    // Combine the children's structure
    auto childrenStruct = Union( myLowerStruct, theirLowerStruct );

    // Now add in the original lower structure
    auto partialStruct = Union( childrenStruct, node.origLowerStruct );

    // Now the node indices
    vector<Int> nodeInds( node.size );
    for( Int i=0; i<node.size; ++i )
        nodeInds[i] = node.off + i;
    auto fullStruct = Union( nodeInds, partialStruct );

    // Construct the relative indices of the original lower structure
    node.origLowerRelInds = RelativeIndices( node.origLowerStruct, fullStruct );

    // Construct the relative indices of the children
    node.childSizes.resize(2);
    node.childRelInds.resize(2);
    if( node.child->onLeft )
    {
        node.childSizes[0] = node.child->size;
        node.childSizes[1] = theirSize;
        node.childRelInds[0] = RelativeIndices( myLowerStruct, fullStruct );
        node.childRelInds[1] = RelativeIndices( theirLowerStruct, fullStruct );
    }
    else
    {
        node.childSizes[0] = theirSize;
        node.childSizes[1] = node.child->size;
        node.childRelInds[0] = RelativeIndices( theirLowerStruct, fullStruct );
        node.childRelInds[1] = RelativeIndices( myLowerStruct, fullStruct );
    }

    // Form lower structure of this node by removing the node indices
    const Int lowerStructSize = fullStruct.size() - node.size;
    node.lowerStruct.resize( lowerStructSize );
    for( Int i=0; i<lowerStructSize; ++i )
        node.lowerStruct[i] = fullStruct[node.size+i];
    DEBUG_ONLY(
        // Ensure that the root process computed a lowerStruct of the same size
        Int rootLowerStructSize;
        if( mpi::Rank(node.comm) == 0 )
            rootLowerStructSize = lowerStructSize;
        mpi::Broadcast( &rootLowerStructSize, 1, 0, node.comm );
        if( rootLowerStructSize != lowerStructSize )
            RuntimeError("Root has different lower struct size");
    )
}

Int Analysis( NodeInfo& node, Int myOff )
{
    DEBUG_ONLY(CSE cse("ldl::Analysis"))

    // Recurse on the children
    // NOTE: Cleanup of existing info children should be added
    const Int numChildren = node.children.size();
    for( Int c=0; c<numChildren; ++c )
    {
        if( node.children[c] == nullptr )
            LogicError("Node child ",c," was nullptr");
        myOff = Analysis( *node.children[c], myOff );
    }
    
    DEBUG_ONLY(
      if( !IsStrictlySorted(node.origLowerStruct) )
      {
          if( IsSorted(node.origLowerStruct) )
              LogicError("Repeat in original lower struct");
          else
              LogicError("Original lower struct not sorted");
      }
    )

    if( numChildren > 0 )
    {
        // Union the structures of the children with the original structure
        auto fullStruct = node.origLowerStruct;
        for( NodeInfo* child : node.children )
        {
            DEBUG_ONLY(
                if( !IsStrictlySorted(child->lowerStruct) )
                {
                    if( IsSorted(child->lowerStruct) )
                        LogicError("Repeat in child lower struct");
                    else
                        LogicError("Child lower struct not sorted");
                }
            )

            // Combine the structures of the children
            fullStruct = Union( fullStruct, child->lowerStruct );
        }

        // Now add in the node indices
        vector<Int> nodeInds( node.size );
        for( Int i=0; i<node.size; ++i )
            nodeInds[i] = node.off+ i;
        fullStruct = Union( fullStruct, nodeInds );

        // Construct the relative indices of the original lower structure
        node.origLowerRelInds = 
          RelativeIndices( node.origLowerStruct, fullStruct );

        // Construct the relative indices of the children
        node.childRelInds.resize( numChildren );
        for( Int c=0; c<numChildren; ++c )
            node.childRelInds[c] = 
                RelativeIndices( node.children[c]->lowerStruct, fullStruct );

        // Form lower struct of this node by removing node indices
        // (which take up the first node.size indices of fullStruct)
        const Int lowerStructSize = fullStruct.size()-node.size;
        node.lowerStruct.resize( lowerStructSize );
        for( Int i=0; i<lowerStructSize; ++i )
            node.lowerStruct[i] = fullStruct[node.size+i];
    }
    else
    {
        node.lowerStruct = node.origLowerStruct;

        // Construct the trivial relative indices of the original structure
        const Int numOrigLowerInds = node.origLowerStruct.size();
        node.origLowerRelInds.resize( numOrigLowerInds );
        for( Int i=0; i<numOrigLowerInds; ++i )
            node.origLowerRelInds[i] = i + node.size;
    }

    return myOff + node.size;
}

//
// This is the part of the analysis that requires fine-grain parallelism.
// For now, we will assume that the distributed part of the elimination 
// tree is binary.
//

void Analysis( DistNodeInfo& node, bool computeFactRecvInds )
{
    DEBUG_ONLY(CSE cse("ldl::Analysis"))

    node.grid = new Grid( node.comm );

    if( node.duplicate != nullptr )
    {
        Int myOff = 0;
        auto& dupNode = *node.duplicate;
        Analysis( dupNode, myOff );

        // The bottom node was analyzed locally, so just copy its results over
        node.myOff = dupNode.myOff;
        node.lowerStruct = dupNode.lowerStruct;
        node.origLowerRelInds = dupNode.origLowerRelInds;
        node.childRelInds = dupNode.childRelInds;

        const Int numChildren = dupNode.childRelInds.size();
        node.childSizes.resize( numChildren );
        for( Int c=0; c<numChildren; ++c )
            node.childSizes[c] = dupNode.children[c]->size;

        return;
    }

    if( node.child == nullptr )
        LogicError("Node child was nullptr");
    auto& childNode = *node.child;
    Analysis( childNode, computeFactRecvInds );

    node.myOff = childNode.myOff + childNode.size;

    // Get the lower struct for the child we do not share
    Int theirSize;
    vector<Int> theirLowerStruct;
    GetLowerStruct( theirSize, theirLowerStruct, node );

    // Perform one level of symbolic factorization and then compute
    // a wide variety of relative indices
    ComputeStructAndRelInds( theirSize, theirLowerStruct, node );
}

} // namespace ldl
} // namespace El
