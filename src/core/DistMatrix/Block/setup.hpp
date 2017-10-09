/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/

#include "El/blas_like/level1/Copy/internal_impl.hpp"

namespace El {

#define DM DistMatrix<T,COLDIST,ROWDIST>
#define BDM DistMatrix<T,COLDIST,ROWDIST,BLOCK>
#define BCM BlockMatrix<T>
#define ADM AbstractDistMatrix<T>

// Public section
// ##############

// Constructors and destructors
// ============================

template<typename T>
BDM::DistMatrix( const El::Grid& g, int root )
: BCM(g,root)
{
    if( COLDIST != CIRC || ROWDIST != CIRC )
        this->Matrix().FixSize();
    this->SetShifts();
}

template<typename T>
BDM::DistMatrix
( const El::Grid& g, Int blockHeight, Int blockWidth, int root )
: BCM(g,blockHeight,blockWidth,root)
{
    if( COLDIST != CIRC || ROWDIST != CIRC )
        this->Matrix().FixSize();
    this->SetShifts();
}

template<typename T>
BDM::DistMatrix
( Int height, Int width, const El::Grid& g, int root )
: BCM(g,root)
{
    if( COLDIST != CIRC || ROWDIST != CIRC )
        this->Matrix().FixSize();
    this->SetShifts(); this->Resize(height,width);
}

template<typename T>
BDM::DistMatrix
( Int height, Int width, const El::Grid& g,
  Int blockHeight, Int blockWidth, int root )
: BCM(g,blockHeight,blockWidth,root)
{
    if( COLDIST != CIRC || ROWDIST != CIRC )
        this->Matrix().FixSize();
    this->SetShifts();
    this->Resize(height,width);
}

template<typename T>
BDM::DistMatrix( const BDM& A )
: BCM(A.Grid())
{
    EL_DEBUG_CSE
    if( COLDIST != CIRC || ROWDIST != CIRC )
        this->Matrix().FixSize();
    this->SetShifts();
    if( &A != this )
        *this = A;
    else
        LogicError("Tried to construct block DistMatrix with itself");
}

template<typename T>
template<Dist U,Dist V>
BDM::DistMatrix( const DistMatrix<T,U,V,BLOCK>& A )
: BCM(A.Grid())
{
    EL_DEBUG_CSE
    if( COLDIST != CIRC || ROWDIST != CIRC )
        this->Matrix().FixSize();
    this->SetShifts();
    if( COLDIST != U || ROWDIST != V ||
        reinterpret_cast<const BDM*>(&A) != this )
        *this = A;
    else
        LogicError("Tried to construct block DistMatrix with itself");
}

template<typename T>
BDM::DistMatrix( const AbstractDistMatrix<T>& A )
: BCM(A.Grid())
{
    EL_DEBUG_CSE
    if( COLDIST != CIRC || ROWDIST != CIRC )
        this->Matrix().FixSize();
    this->SetShifts();
    #define GUARD(CDIST,RDIST,WRAP) \
      A.ColDist() == CDIST && A.RowDist() == RDIST && A.Wrap() == WRAP
    #define PAYLOAD(CDIST,RDIST,WRAP) \
      auto& ACast = \
        static_cast<const DistMatrix<T,CDIST,RDIST,WRAP>&>(A); \
      *this = ACast;
    #include "El/macros/GuardAndPayload.h"
}

template<typename T>
BDM::DistMatrix( const BlockMatrix<T>& A )
: BCM(A.Grid())
{
    EL_DEBUG_CSE
    if( COLDIST != CIRC || ROWDIST != CIRC )
        this->Matrix().FixSize();
    this->SetShifts();
    #define GUARD(CDIST,RDIST,WRAP) \
      A.DistData().colDist == CDIST && A.DistData().rowDist == RDIST && \
      A.Wrap() == WRAP
    #define PAYLOAD(CDIST,RDIST,WRAP) \
      auto& ACast = \
        static_cast<const DistMatrix<T,CDIST,RDIST,BLOCK>&>(A); \
      if( COLDIST != CDIST || ROWDIST != RDIST || BLOCK != WRAP || \
          reinterpret_cast<const BDM*>(&A) != this ) \
          *this = ACast; \
      else \
          LogicError("Tried to construct DistMatrix with itself");
    #include "El/macros/GuardAndPayload.h"
}

template<typename T>
template<Dist U,Dist V>
BDM::DistMatrix( const DistMatrix<T,U,V>& A )
: BCM(A.Grid())
{
    EL_DEBUG_CSE
    if( COLDIST != CIRC || ROWDIST != CIRC )
        this->Matrix().FixSize();
    this->SetShifts();
    *this = A;
}

template<typename T>
BDM::DistMatrix( BDM&& A ) EL_NO_EXCEPT : BCM(std::move(A)) { }

template<typename T> BDM::~DistMatrix() { }

template<typename T>
BDM* BDM::Copy() const
{ return new DistMatrix<T,COLDIST,ROWDIST,BLOCK>(*this); }

template<typename T>
BDM* BDM::Construct( const El::Grid& g, int root ) const
{ return new DistMatrix<T,COLDIST,ROWDIST,BLOCK>(g,root); }

template<typename T>
DistMatrix<T,ROWDIST,COLDIST,BLOCK>* BDM::ConstructTranspose
( const El::Grid& g, int root ) const
{ return new DistMatrix<T,ROWDIST,COLDIST,BLOCK>(g,root); }

template<typename T>
typename BDM::diagType*
BDM::ConstructDiagonal
( const El::Grid& g, int root ) const
{ return new DistMatrix<T,DiagCol<COLDIST,ROWDIST>(),
                          DiagRow<COLDIST,ROWDIST>(),BLOCK>(g,root); }

// Operator overloading
// ====================

// Return a view
// -------------
template<typename T>
BDM BDM::operator()( Range<Int> I, Range<Int> J )
{
    EL_DEBUG_CSE
    if( this->Locked() )
        return LockedView( *this, I, J );
    else
        return View( *this, I, J );
}

template<typename T>
const BDM BDM::operator()( Range<Int> I, Range<Int> J ) const
{
    EL_DEBUG_CSE
    return LockedView( *this, I, J );
}

// Non-contiguous
// --------------
template<typename T>
BDM BDM::operator()( Range<Int> I, const vector<Int>& J ) const
{
    EL_DEBUG_CSE
    BDM ASub( this->Grid(), this->BlockHeight(), this->BlockWidth() );
    GetSubmatrix( *this, I, J, ASub );
    return ASub;
}

template<typename T>
BDM BDM::operator()( const vector<Int>& I, Range<Int> J ) const
{
    EL_DEBUG_CSE
    BDM ASub( this->Grid(), this->BlockHeight(), this->BlockWidth() );
    GetSubmatrix( *this, I, J, ASub );
    return ASub;
}

template<typename T>
BDM BDM::operator()( const vector<Int>& I, const vector<Int>& J ) const
{
    EL_DEBUG_CSE
    BDM ASub( this->Grid(), this->BlockHeight(), this->BlockWidth() );
    GetSubmatrix( *this, I, J, ASub );
    return ASub;
}

// Copy
// ----

template<typename T>
BDM& BDM::operator=( const AbstractDistMatrix<T>& A )
{
    EL_DEBUG_CSE
    #define GUARD(CDIST,RDIST,WRAP) \
      A.ColDist() == CDIST && A.RowDist() == RDIST && A.Wrap() == WRAP
    #define PAYLOAD(CDIST,RDIST,WRAP) \
      auto& ACast = \
        static_cast<const DistMatrix<T,CDIST,RDIST,WRAP>&>(A); \
      *this = ACast;
    #include "El/macros/GuardAndPayload.h"
    return *this;
}

template<typename T>
template<Dist U,Dist V>
BDM& BDM::operator=( const DistMatrix<T,U,V>& A )
{
    EL_DEBUG_CSE
    // TODO: Use either AllGather or Gather if the distribution of this matrix
    //       is respectively either (STAR,STAR) or (CIRC,CIRC)
    // TODO: Specially handle cases where the block size is 1 x 1
    copy::GeneralPurpose( A, *this );
    return *this;
}

template<typename T>
BDM& BDM::operator=( BDM&& A )
{
    if( this->Viewing() || A.Viewing() )
        this->operator=( (const BDM&)A );
    else
        BCM::operator=( std::move(A) );
    return *this;
}

// Rescaling
// ---------
template<typename T>
const BDM& BDM::operator*=( T alpha )
{
    EL_DEBUG_CSE
    Scale( alpha, *this );
    return *this;
}

// Addition/subtraction
// --------------------
template<typename T>
const BDM& BDM::operator+=( const BCM& A )
{
    EL_DEBUG_CSE
    Axpy( T(1), A, *this );
    return *this;
}

template<typename T>
const BDM& BDM::operator+=( const ADM& A )
{
    EL_DEBUG_CSE
    Axpy( T(1), A, *this );
    return *this;
}

template<typename T>
const BDM& BDM::operator-=( const BCM& A )
{
    EL_DEBUG_CSE
    Axpy( T(-1), A, *this );
    return *this;
}

template<typename T>
const BDM& BDM::operator-=( const ADM& A )
{
    EL_DEBUG_CSE
    Axpy( T(-1), A, *this );
    return *this;
}

// Distribution data
// =================
template<typename T>
Dist BDM::ColDist() const EL_NO_EXCEPT { return COLDIST; }
template<typename T>
Dist BDM::RowDist() const EL_NO_EXCEPT { return ROWDIST; }

template<typename T>
Dist BDM::PartialColDist() const EL_NO_EXCEPT { return Partial<COLDIST>(); }
template<typename T>
Dist BDM::PartialRowDist() const EL_NO_EXCEPT { return Partial<ROWDIST>(); }

template<typename T>
Dist BDM::PartialUnionColDist() const EL_NO_EXCEPT
{ return PartialUnionCol<COLDIST,ROWDIST>(); }
template<typename T>
Dist BDM::PartialUnionRowDist() const EL_NO_EXCEPT
{ return PartialUnionRow<COLDIST,ROWDIST>(); }

template<typename T>
Dist BDM::CollectedColDist() const EL_NO_EXCEPT { return Collect<COLDIST>(); }
template<typename T>
Dist BDM::CollectedRowDist() const EL_NO_EXCEPT { return Collect<ROWDIST>(); }

} // namespace El
