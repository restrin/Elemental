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

template<typename T>
DistMultiVecNode<T>::DistMultiVecNode( DistMultiVecNode<T>* parentNode )
: parent(parentNode), child(nullptr), duplicate(nullptr)
{ }

template<typename T>
DistMultiVecNode<T>::DistMultiVecNode
( const DistMap& invMap,
  const DistNodeInfo& info,
  const DistMultiVec<T>& X )
: parent(nullptr), child(nullptr), duplicate(nullptr)
{
    DEBUG_ONLY(CSE cse("DistMultiVecNode::DistMultiVecNode"))
    Pull( invMap, info, X );
}

template<typename T>
DistMultiVecNode<T>::DistMultiVecNode( const DistMatrixNode<T>& X )
: parent(nullptr), child(nullptr), duplicate(nullptr)
{
    DEBUG_ONLY(CSE cse("DistMultiVecNode::DistMultiVecNode"))
    *this = X;
}

template<typename T>
DistMultiVecNode<T>::~DistMultiVecNode()
{
    delete child;
    delete duplicate;
}

template<typename T>
const DistMultiVecNode<T>&
DistMultiVecNode<T>::operator=( const DistMatrixNode<T>& X )
{
    DEBUG_ONLY(CSE cse("DistMultiVecNode::operator="))

    if( X.child == nullptr )
    {
        delete duplicate; 
        duplicate = new MatrixNode<T>(this);
        *duplicate = *X.duplicate;

        matrix.Attach( X.matrix.Grid(), duplicate->matrix );

        return *this;
    }

    matrix.SetGrid( X.matrix.Grid() );
    matrix = X.matrix;

    delete child;
    child = new DistMultiVecNode<T>(this);
    *child = *X.child;

    return *this;
}

// TODO: Consider passing in a pre-built origOwners
template<typename T>
void DistMultiVecNode<T>::Pull
( const DistMap& invMap,
  const DistNodeInfo& info,
  const DistMultiVec<T>& X )
{
    DEBUG_ONLY(CSE cse("DistMultiVecNode::Pull"))
    DistMultiVecNodeMeta meta;
    Pull( invMap, info, X, meta );
}

template<typename T>
static void PullLocalInit
( const NodeInfo& node,
        MatrixNode<T>& XNode,
  Int width )
{
    DEBUG_ONLY(CSE cse("PullLocalInit [DistMultiVecNode]"))
    const Int numChildren = node.children.size();
    if( XNode.children.size() != node.children.size() )
    {
        for( auto* childNode : XNode.children )
            delete childNode;
        XNode.children.resize(numChildren);
        for( Int c=0; c<numChildren; ++c )
            XNode.children[c] = new MatrixNode<T>(&XNode);
    }

    for( Int c=0; c<numChildren; ++c )
        PullLocalInit( *node.children[c], *XNode.children[c], width );

    XNode.matrix.Resize( node.size, width );
}

template<typename T>
static void PullInit
( const DistNodeInfo& node,
        DistMultiVecNode<T>& XNode,
  Int width )
{
    DEBUG_ONLY(CSE cse("PullInit [DistMultiVecNode]"))
    if( node.child == nullptr )
    {
        DEBUG_ONLY(
          if( XNode.child != nullptr )
              LogicError("Child should have been a nullptr");
        )
        if( XNode.duplicate == nullptr )
            XNode.duplicate = new MatrixNode<T>(&XNode);
        PullLocalInit( *node.duplicate, *XNode.duplicate, width );

        XNode.matrix.Attach( *node.grid, XNode.duplicate->matrix );
        return;
    }

    DEBUG_ONLY(
      if( XNode.duplicate != nullptr )
          LogicError("Duplicate should have been a nullptr");
    )
    if( XNode.child == nullptr )
        XNode.child = new DistMultiVecNode<T>(&XNode);
    PullInit( *node.child, *XNode.child, width );

    if( XNode.matrix.Grid() != *node.grid ||
        XNode.matrix.Height() != node.size ||
        XNode.matrix.Width() != width )
    {
        XNode.commMeta.Empty();
        XNode.matrix.SetGrid( *node.grid );
        XNode.matrix.Resize( node.size, width );
    }
}

template<typename T>
static void PullLocalUnpack
( const NodeInfo& info,
  const vector<T>& recvVals,
  const vector<int>& mappedOwners,
        MatrixNode<T>& XNode,
  Int& off, std::vector<int>& offs )
{
    DEBUG_ONLY(CSE cse("PullLocalUnpack [DistMultiVecNode]"))
    const Int numChildren = info.children.size();
    for( Int c=0; c<numChildren; ++c )
        PullLocalUnpack
        ( *info.children[c], recvVals, mappedOwners, *XNode.children[c],
          off, offs );

    T* XNodeBuf = XNode.matrix.Buffer();
    for( Int t=0; t<info.size; ++t )
    {
        const int q = mappedOwners[off++];
        XNodeBuf[t] = recvVals[offs[q]++];
    }
}

template<typename T>
static void PullLocalUnpackMulti
( const NodeInfo& info,
  const vector<T>& recvVals,
  const vector<int>& mappedOwners,
        MatrixNode<T>& XNode,
  Int& off, std::vector<int>& offs,
  Int width )
{
    DEBUG_ONLY(CSE cse("PullLocalUnpackMulti [DistMultiVecNode]"))
    const Int numChildren = info.children.size();
    for( Int c=0; c<numChildren; ++c )
        PullLocalUnpackMulti
        ( *info.children[c], recvVals, mappedOwners, *XNode.children[c],
          off, offs, width );

    T* XNodeBuf = XNode.matrix.Buffer();
    const Int XNodeLDim = XNode.matrix.LDim();
    for( Int t=0; t<info.size; ++t )
    {
        const int q = mappedOwners[off++];
        for( Int j=0; j<width; ++j )
            XNodeBuf[t+j*XNodeLDim] = recvVals[offs[q]++];
    }
}

template<typename T>
static void PullUnpack
( const DistNodeInfo& info,
  const vector<T>& recvVals,
  const vector<int>& mappedOwners,
        DistMultiVecNode<T>& XNode,
  Int& off, std::vector<int>& offs )
{
    if( info.child == nullptr )
    {
        PullLocalUnpack
        ( *info.duplicate, recvVals, mappedOwners, 
          *XNode.duplicate, off, offs );
        return;
    }
    PullUnpack( *info.child, recvVals, mappedOwners, *XNode.child, off, offs );

    const Int localHeight = XNode.matrix.LocalHeight();
    T* XNodeBuf = XNode.matrix.Buffer();
    for( Int tLoc=0; tLoc<localHeight; ++tLoc )
    {
        const int q = mappedOwners[off++];
        XNodeBuf[tLoc] = recvVals[offs[q]++];
    }
}

template<typename T>
static void PullUnpackMulti
( const DistNodeInfo& info,
  const vector<T>& recvVals,
  const vector<int>& mappedOwners,
        DistMultiVecNode<T>& XNode,
  Int& off, std::vector<int>& offs,
  Int width )
{
    if( info.child == nullptr )
    {
        PullLocalUnpackMulti
        ( *info.duplicate, recvVals, mappedOwners, 
          *XNode.duplicate, off, offs, width );
        return;
    }
    PullUnpackMulti
    ( *info.child, recvVals, mappedOwners, *XNode.child, off, offs, width );

    const Int localHeight = XNode.matrix.LocalHeight();
    T* XNodeBuf = XNode.matrix.Buffer();
    const Int XNodeLDim = XNode.matrix.LDim();
    for( Int tLoc=0; tLoc<localHeight; ++tLoc )
    {
        const int q = mappedOwners[off++];
        for( Int j=0; j<width; ++j )
            XNodeBuf[tLoc+j*XNodeLDim] = recvVals[offs[q]++];
    }
}

template<typename T>
void DistMultiVecNode<T>::Pull
( const DistMap& invMap,
  const DistNodeInfo& info,
  const DistMultiVec<T>& X,
        DistMultiVecNodeMeta& meta )
{
    DEBUG_ONLY(CSE cse("DistMultiVecNode::Pull"))
    const Int width = X.Width();
    const T* XBuf = X.LockedMatrix().LockedBuffer();
    const Int XLDim = X.LockedMatrix().LDim();
    const Int firstLocalRow = X.FirstLocalRow();
    mpi::Comm comm = X.Comm();
    const int commSize = mpi::Size( comm );

    PullInit( info, *this, width );

    // NOTE: send/recv are reversed for Push and Pull; we use Push as reference,
    //       so they will be reversed in this routine
    meta.Initialize( *this, info, invMap, X );

    // Fulfill the requests
    const Int numSendInds = meta.recvInds.size();
    vector<T> sendVals( numSendInds*width );
    if( width == 1 )
    {
        for( Int s=0; s<numSendInds; ++s )
            sendVals[s] = XBuf[meta.recvInds[s]-firstLocalRow];
    }
    else
    {
        for( Int s=0; s<numSendInds; ++s )
            for( Int j=0; j<width; ++j )
                sendVals[s*width+j] = 
                  XBuf[meta.recvInds[s]-firstLocalRow+j*XLDim];
    }

    // Reply with the values
    const Int numRecvInds = meta.mappedOwners.size();
    vector<T> recvVals( numRecvInds*width );
    if( width != 1 )
    {
        for( int q=0; q<commSize; ++q )
        {
            meta.sendSizes[q] *= width;
            meta.sendOffs[q] *= width;
            meta.recvSizes[q] *= width;
            meta.recvOffs[q] *= width;
        }
    }
    mpi::AllToAll
    ( sendVals.data(), meta.recvSizes.data(), meta.recvOffs.data(),
      recvVals.data(), meta.sendSizes.data(), meta.sendOffs.data(), comm );
    SwapClear( sendVals );

    // Unpack the values
    Int off = 0;
    auto offs = meta.sendOffs;
    if( width == 1 )
    {
        PullUnpack
        ( info, recvVals, meta.mappedOwners, *this, off, offs );
    }
    else
    {
        PullUnpackMulti
        ( info, recvVals, meta.mappedOwners, *this, off, offs, width );
        for( int q=0; q<commSize; ++q )
        {
            meta.sendSizes[q] /= width;
            meta.sendOffs[q] /= width;
            meta.recvSizes[q] /= width;
            meta.recvOffs[q] /= width;
        }
    }
    DEBUG_ONLY(
      if( off != numRecvInds )
          LogicError("Unpacked wrong number of indices");
    )
}

template<typename T>
static void PushLocalPack
( const NodeInfo& node,
  const MatrixNode<T>& XNode,
  const vector<int>& mappedOwners, 
        vector<T>& sendVals,
  Int& off, vector<int>& offs )
{
    DEBUG_ONLY(CSE cse("PushLocalPack [DistMultiVecNode]"))
    const Int numChildren = node.children.size();
    for( Int c=0; c<numChildren; ++c )
        PushLocalPack
        ( *node.children[c], *XNode.children[c], mappedOwners,
          sendVals, off, offs );

    const T* XNodeBuf = XNode.matrix.LockedBuffer();
    for( Int t=0; t<node.size; ++t )
    {
        const int q = mappedOwners[off++];
        sendVals[offs[q]++] = XNodeBuf[t];
    }
}

template<typename T>
static void PushLocalPackMulti
( const NodeInfo& node,
  const MatrixNode<T>& XNode,
  const vector<int>& mappedOwners, 
        vector<T>& sendVals,
  Int& off, vector<int>& offs )
{
    DEBUG_ONLY(CSE cse("PushLocalPackMulti [DistMultiVecNode]"))
    const Int numChildren = node.children.size();
    for( Int c=0; c<numChildren; ++c )
        PushLocalPackMulti
        ( *node.children[c], *XNode.children[c], mappedOwners,
          sendVals, off, offs );

    const T* XNodeBuf = XNode.matrix.LockedBuffer();
    const Int XNodeLDim = XNode.matrix.LDim();
    const Int width = XNode.matrix.Width();
    for( Int t=0; t<node.size; ++t )
    {
        const int q = mappedOwners[off++];
        for( Int j=0; j<width; ++j )
            sendVals[offs[q]*width+j] = XNodeBuf[t+j*XNodeLDim];
        offs[q]++; 
    }
}

template<typename T>
static void PushPack
( const DistNodeInfo& node,
  const DistMultiVecNode<T>& XNode,
  const vector<int>& mappedOwners,
        vector<T>& sendVals,
  Int& off, vector<int>& offs )
{
    DEBUG_ONLY(CSE cse("PushPack [DistMultiVecNode]"))
    if( node.child == nullptr )
    {
        PushLocalPack
        ( *node.duplicate, *XNode.duplicate,
          mappedOwners, sendVals, off, offs );
        return;
    }
    PushPack( *node.child, *XNode.child, mappedOwners, sendVals, off, offs );
            
    const Int localHeight = XNode.matrix.LocalHeight();
    const T* XNodeBuf = XNode.matrix.LockedBuffer();
    for( Int tLoc=0; tLoc<localHeight; ++tLoc )
    {
        const int q = mappedOwners[off++];
        sendVals[offs[q]++] = XNodeBuf[tLoc];
    }
}

template<typename T>
static void PushPackMulti
( const DistNodeInfo& node,
  const DistMultiVecNode<T>& XNode,
  const vector<int>& mappedOwners,
        vector<T>& sendVals,
  Int& off, vector<int>& offs )
{
    DEBUG_ONLY(CSE cse("PushPackMulti [DistMultiVecNode]"))
    if( node.child == nullptr )
    {
        PushLocalPackMulti
        ( *node.duplicate, *XNode.duplicate,
          mappedOwners, sendVals, off, offs );
        return;
    }
    PushPackMulti
    ( *node.child, *XNode.child, mappedOwners, sendVals, off, offs );
    
    const Int localHeight = XNode.matrix.LocalHeight();
    const T* XNodeBuf = XNode.matrix.LockedBuffer();
    const Int XNodeLDim = XNode.matrix.LDim();
    const Int width = XNode.matrix.Width();
    for( Int tLoc=0; tLoc<localHeight; ++tLoc )
    {
        const int q = mappedOwners[off++];
        for( Int j=0; j<width; ++j )
            sendVals[offs[q]*width+j] = XNodeBuf[tLoc+j*XNodeLDim];
        offs[q]++;
    }
}

template<typename T>
void DistMultiVecNode<T>::Push
( const DistMap& invMap,
  const DistNodeInfo& info,
        DistMultiVec<T>& X ) const
{
    DEBUG_ONLY(CSE cse("DistMultiVecNode::Push"))
    DistMultiVecNodeMeta meta;
    Push( invMap, info, X, meta );
}

template<typename T>
void DistMultiVecNodeMeta::Initialize
( const DistMultiVecNode<T>& XNode,
  const DistNodeInfo& info,
  const DistMap& invMap,
  const DistMultiVec<T>& X )
{
    DEBUG_ONLY(CSE cse("DistMultiVecNodeMeta::Initialize"))
    if( sendInds.size()     != 0 &&
        recvInds.size()     != 0 &&
        mappedOwners.size() != 0 &&
        sendSizes.size()    != 0 &&
        sendOffs.size()     != 0 &&
        recvSizes.size()    != 0 &&
        recvOffs.size()     != 0 )
    {
        return;
    }

    const Int numSendInds = XNode.LocalHeight();
    mpi::Comm comm = info.comm;
    const int commSize = mpi::Size( comm );

    std::vector<Int> mappedInds( numSendInds );
    Int off=0;
    function<void(const NodeInfo&,MatrixNode<T>&)> localPack = 
      [&]( const NodeInfo& node, MatrixNode<T>& XNode )
      {
        const Int numChildren = node.children.size();
        for( Int c=0; c<numChildren; ++c )
            localPack( *node.children[c], *XNode.children[c] );

        for( Int t=0; t<node.size; ++t )
            mappedInds[off++] = node.off+t;
      };
    function<void(const DistNodeInfo&,const DistMultiVecNode<T>&)> 
      pack = 
      [&]( const DistNodeInfo& node, const DistMultiVecNode<T>& XNode )
      {
        if( node.child == nullptr )
        {
            localPack( *node.duplicate, *XNode.duplicate );
            return;
        }
        pack( *node.child, *XNode.child );

        const Int nodeHeight = XNode.matrix.Height();
        const Int colShift = XNode.matrix.ColShift();
        const Int colStride = XNode.matrix.ColStride();
        for( Int t=colShift; t<nodeHeight; t+=colStride )
            mappedInds[off++] = node.off+t;
      };
    pack( info, XNode );

    // Convert the indices to the original ordering
    // TODO: Consider passing in a pre-built origOwners
    invMap.Translate( mappedInds );

    mappedOwners.resize( numSendInds );
    for( Int s=0; s<numSendInds; ++s )
        mappedOwners[s] = X.RowOwner(mappedInds[s]);

    // Figure out how many indices each process owns that we need to send
    sendSizes.resize( commSize );
    for( int q=0; q<commSize; ++q )
        sendSizes[q] = 0;
    for( Int s=0; s<numSendInds; ++s )
        ++sendSizes[mappedOwners[s]];
    Scan( sendSizes, sendOffs );

    // Pack the send indices
    sendInds.resize( numSendInds );
    off=0;
    auto offs = sendOffs;

    function<void(const NodeInfo&,const MatrixNode<T>&)> localPackInds =
      [&]( const NodeInfo& node, const MatrixNode<T>& XNode )
      {
        const Int numChildren = node.children.size();
        for( Int c=0; c<numChildren; ++c )
            localPackInds( *node.children[c], *XNode.children[c] );

        for( Int t=0; t<node.size; ++t )
        {
            const Int i = mappedInds[off];
            const int q = mappedOwners[off++];
            sendInds[offs[q]++] = i;
        }
      };
    function<void(const DistNodeInfo&,const DistMultiVecNode<T>&)> 
      packInds =
      [&]( const DistNodeInfo& node, const DistMultiVecNode<T>& XNode )
      {
        if( node.child == nullptr )
        {
            localPackInds( *node.duplicate, *XNode.duplicate );
            return;
        }
        packInds( *node.child, *XNode.child );
    
        const Int localHeight = XNode.matrix.LocalHeight();
        for( Int tLoc=0; tLoc<localHeight; ++tLoc )
        {
            const Int i = mappedInds[off];
            const int q = mappedOwners[off++];
            sendInds[offs[q]++] = i;
        }
      };
    packInds( info, XNode );

    // Coordinate for the coming AllToAll to exchange the indices of x
    recvSizes.resize( commSize );
    mpi::AllToAll( sendSizes.data(), 1, recvSizes.data(), 1, comm );
    const int numRecvInds = Scan( recvSizes, recvOffs );
    DEBUG_ONLY(
      if( numRecvInds != X.LocalHeight() )
          LogicError("numRecvInds was not equal to local height");
    )

    // Send the indices
    recvInds.resize( numRecvInds );
    mpi::AllToAll
    ( sendInds.data(), sendSizes.data(), sendOffs.data(),
      recvInds.data(), recvSizes.data(), recvOffs.data(), comm );
}

template<typename T>
void DistMultiVecNode<T>::Push
( const DistMap& invMap,
  const DistNodeInfo& info,
        DistMultiVec<T>& X,
        DistMultiVecNodeMeta& meta ) const
{
    DEBUG_ONLY(CSE cse("DistMultiVecNode::Push"))
    const Int height = info.size + info.off;
    const Int width = matrix.Width();
    Timer timer;
    bool time = false;

    mpi::Comm comm = info.comm;
    const int commSize = mpi::Size( comm );
    const int commRank = mpi::Rank( comm );
    X.SetComm( comm );
    X.Resize( height, width );
    const Int firstLocalRow = X.FirstLocalRow();
    T* XBuf = X.Matrix().Buffer();
    const Int XLDim = X.Matrix().LDim();

    meta.Initialize( *this, info, invMap, X );
    const int numSendInds = meta.mappedOwners.size();

    // Pack the send values
    if( time && commRank == 0 )
        timer.Start();
    const Int numRecvInds = meta.recvInds.size();
    vector<T> sendVals( numSendInds*width );
    {
        Int off=0;
        auto offs = meta.sendOffs;
        if( width == 1 )
            PushPack
            ( info, *this, meta.mappedOwners, sendVals, off, offs );
        else
            PushPackMulti
            ( info, *this, meta.mappedOwners, sendVals, off, offs );
    }
    if( time && commRank == 0 )
        Output("  pack time: ",timer.Stop()," secs");

    // Send the values
    if( time && commRank == 0 )
        timer.Start();
    vector<T> recvVals( numRecvInds*width );
    if( width != 1 )
    {
        for( int q=0; q<commSize; ++q )
        {
            meta.sendSizes[q] *= width;
            meta.sendOffs[q] *= width;
            meta.recvSizes[q] *= width;
            meta.recvOffs[q] *= width;
        }
    }
    mpi::AllToAll
    ( sendVals.data(), meta.sendSizes.data(), meta.sendOffs.data(),
      recvVals.data(), meta.recvSizes.data(), meta.recvOffs.data(), comm );
    SwapClear( sendVals );
    if( time && commRank == 0 )
        Output("  send time: ",timer.Stop()," secs");

    // Unpack the values
    if( time && commRank == 0 )
        timer.Start();
    if( width == 1 )
    {
        for( Int s=0; s<numRecvInds; ++s )
        {
            const Int i = meta.recvInds[s];
            const Int iLoc = i - firstLocalRow;
            XBuf[iLoc] = recvVals[s];
        }
    }
    else
    {
        for( Int s=0; s<numRecvInds; ++s )
        {
            const Int i = meta.recvInds[s];
            const Int iLoc = i - firstLocalRow;
            for( Int j=0; j<width; ++j )
                XBuf[iLoc+j*XLDim] = recvVals[s*width+j];
        }
        for( int q=0; q<commSize; ++q )
        {
            meta.sendSizes[q] /= width;
            meta.sendOffs[q] /= width;
            meta.recvSizes[q] /= width;
            meta.recvOffs[q] /= width;
        }
    }
    if( time && commRank == 0 )
        Output("  unpack time: ",timer.Stop()," secs");
}

template<typename T>
Int DistMultiVecNode<T>::LocalHeight() const
{
    DEBUG_ONLY(CSE cse("DistMultiVecNode::LocalHeight"))
    Int localHeight = 0;
    function<void(const DistMultiVecNode<T>&)> count = 
      [&]( const DistMultiVecNode<T>& node )
      {
        if( node.child == nullptr )
        {
            localHeight += node.duplicate->Height();
            return;
        }
        count( *node.child );
        localHeight += node.matrix.LocalHeight();
      };
    count( *this );
    return localHeight;
}

template<typename T>
void DistMultiVecNode<T>::ComputeCommMeta( const DistNodeInfo& info ) const
{
    DEBUG_ONLY(CSE cse("DistMultiVecNode::ComputeCommMeta"))
    if( commMeta.numChildSendInds.size() != 0 )
        return;

    commMeta.Empty();
    if( child == nullptr )
    {
        commMeta.localOff = info.duplicate->myOff;
        commMeta.localSize = info.duplicate->size;
        return;
    }

    // This is currently assumed (and will eventually be lifted)
    const Int numChildren = 2;
    vector<int> teamSizes(numChildren), teamOffs(numChildren);

    const int teamSize = mpi::Size( info.comm );
    const int teamRank = mpi::Rank( info.comm );
    const auto& childNode = *info.child;
    const int childTeamSize = mpi::Size( childNode.comm );
    const int childTeamRank = mpi::Rank( childNode.comm );
    const bool inFirstTeam = ( childTeamRank == teamRank );
    const bool leftIsFirst = ( childNode.onLeft==inFirstTeam );
    teamSizes[0] =
        ( childNode.onLeft ? childTeamSize : teamSize-childTeamSize );
    teamSizes[1] = teamSize - teamSizes[0];
    teamOffs[0] = ( leftIsFirst ? 0 : teamSizes[1] );
    teamOffs[1] = ( leftIsFirst ? teamSizes[0] : 0 );

    const auto& myRelInds =
        ( childNode.onLeft ? info.childRelInds[0] : info.childRelInds[1] );

    // Fill numChildSendInds
    commMeta.numChildSendInds.resize( teamSize );
    El::MemZero( commMeta.numChildSendInds.data(), teamSize );
    const Int updateSize = childNode.lowerStruct.size();
    // TODO: Use 'matrix' or 'work' instead?
    {
        const Int align = childNode.size % childTeamSize;
        const Int shift = Shift( childTeamRank, align, childTeamSize );
        const Int localHeight = Length( updateSize, shift, childTeamSize );
        for( Int iChildLoc=0; iChildLoc<localHeight; ++iChildLoc )
        {
            const Int iChild = shift + iChildLoc*childTeamSize;
            const int destRank = myRelInds[iChild] % teamSize;
            ++commMeta.numChildSendInds[destRank];
        }
    }

    // Compute the solve recv indices
    commMeta.childRecvInds.resize( teamSize );
    for( Int c=0; c<numChildren; ++c )
    {
        const Int numInds = info.childRelInds[c].size();
        vector<Int> inds;
        for( Int i=0; i<numInds; ++i )
            if( work.IsLocalRow( info.childRelInds[c][i] ) )
                inds.push_back( i );

        const Int numSolveInds = inds.size();
        for( Int iPre=0; iPre<numSolveInds; ++iPre )
        {
            const Int iChild = inds[iPre];
            const Int i = info.childRelInds[c][iChild];
            const Int iLoc = (i-teamRank) / teamSize;
            const int childRank = (info.childSizes[c]+iChild) % teamSizes[c];
            const int frontRank = teamOffs[c] + childRank;
            commMeta.childRecvInds[frontRank].push_back(iLoc);
        }
    }
    commMeta.localOff = child->commMeta.localOff + child->commMeta.localSize;
    commMeta.localSize = matrix.LocalHeight();
}

#define PROTO(T) template struct DistMultiVecNode<T>;
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace ldl
} // namespace El
