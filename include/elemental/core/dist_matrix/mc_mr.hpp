/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef CORE_DISTMATRIX_MC_MR_DECL_HPP
#define CORE_DISTMATRIX_MC_MR_DECL_HPP

namespace elem {

// Partial specialization to A[MC,MR].
//
// The columns of these matrices will be distributed among columns of the
// process grid, and the rows will be distributed among rows of the process
// grid.

template<typename T,typename Int>
class DistMatrix<T,MC,MR,Int> : public AbstractDistMatrix<T,Int>
{
public:
    // Create a 0 x 0 distributed matrix
    DistMatrix( const elem::Grid& g=DefaultGrid() );

    // Create a height x width distributed matrix
    DistMatrix
    ( Int height, Int width, const elem::Grid& g=DefaultGrid() );

    // Create a height x width distributed matrix with specified alignments
    DistMatrix
    ( Int height, Int width, Int colAlignment, Int rowAlignment, 
      const elem::Grid& g );

    // Create a height x width distributed matrix with specified alignments
    // and leading dimension
    DistMatrix
    ( Int height, Int width, Int colAlignment, Int rowAlignment, 
      Int ldim, const elem::Grid& g );

    // View a constant distributed matrix's buffer
    DistMatrix
    ( Int height, Int width, Int colAlignment, Int rowAlignment,
      const T* buffer, Int ldim, const elem::Grid& g );

    // View a mutable distributed matrix's buffer
    DistMatrix
    ( Int height, Int width, Int colAlignment, Int rowAlignment,
      T* buffer, Int ldim, const elem::Grid& g );

    // Create a copy of distributed matrix A
    template<Distribution U,Distribution V>
    DistMatrix( const DistMatrix<T,U,V,Int>& A );

    ~DistMatrix();

    const DistMatrix<T,MC,MR,Int>& 
    operator=( const DistMatrix<T,MC,MR,Int>& A );

    const DistMatrix<T,MC,MR,Int>& 
    operator=( const DistMatrix<T,MC,STAR,Int>& A );

    const DistMatrix<T,MC,MR,Int>& 
    operator=( const DistMatrix<T,STAR,MR,Int>& A );

    const DistMatrix<T,MC,MR,Int>& 
    operator=( const DistMatrix<T,MD,STAR,Int>& A );

    const DistMatrix<T,MC,MR,Int>& 
    operator=( const DistMatrix<T,STAR,MD,Int>& A );

    const DistMatrix<T,MC,MR,Int>& 
    operator=( const DistMatrix<T,MR,MC,Int>& A );

    const DistMatrix<T,MC,MR,Int>& 
    operator=( const DistMatrix<T,MR,STAR,Int>& A );

    const DistMatrix<T,MC,MR,Int>& 
    operator=( const DistMatrix<T,STAR,MC,Int>& A );

    const DistMatrix<T,MC,MR,Int>& 
    operator=( const DistMatrix<T,VC,STAR,Int>& A );

    const DistMatrix<T,MC,MR,Int>& 
    operator=( const DistMatrix<T,STAR,VC,Int>& A );

    const DistMatrix<T,MC,MR,Int>& 
    operator=( const DistMatrix<T,VR,STAR,Int>& A );

    const DistMatrix<T,MC,MR,Int>& 
    operator=( const DistMatrix<T,STAR,VR,Int>& A );

    const DistMatrix<T,MC,MR,Int>& 
    operator=( const DistMatrix<T,STAR,STAR,Int>& A );

    //------------------------------------------------------------------------//
    // Overrides of AbstractDistMatrix                                        //
    //------------------------------------------------------------------------//

    //
    // Non-collective routines
    //

    virtual Int ColStride() const;
    virtual Int RowStride() const;
    virtual Int ColRank() const;
    virtual Int RowRank() const;
    virtual elem::DistData<Int> DistData() const;

    //
    // Collective routines
    //

    virtual T Get( Int i, Int j ) const;
    virtual void Set( Int i, Int j, T alpha );
    virtual void Update( Int i, Int j, T alpha );

    virtual void ResizeTo( Int height, Int width );

    // Distribution alignment
    virtual void AlignWith( const elem::DistData<Int>& data );
    virtual void AlignWith( const AbstractDistMatrix<T,Int>& A );
    virtual void AlignColsWith( const elem::DistData<Int>& data );
    virtual void AlignColsWith( const AbstractDistMatrix<T,Int>& A );
    virtual void AlignRowsWith( const elem::DistData<Int>& data );
    virtual void AlignRowsWith( const AbstractDistMatrix<T,Int>& A );

    //
    // Though the following routines are meant for complex data, all but two
    // logically apply to real data.
    //

    virtual typename Base<T>::type GetRealPart( Int i, Int j ) const;
    virtual typename Base<T>::type GetImagPart( Int i, Int j ) const;
    virtual void SetRealPart( Int i, Int j, typename Base<T>::type u );
    // Only valid for complex data
    virtual void SetImagPart( Int i, Int j, typename Base<T>::type u );
    virtual void UpdateRealPart( Int i, Int j, typename Base<T>::type u );
    // Only valid for complex data
    virtual void UpdateImagPart( Int i, Int j, typename Base<T>::type u );

    //-----------------------------------------------------------------------//
    // Routines specific to [MC,MR] distribution                             //
    //-----------------------------------------------------------------------//

    //
    // Collective routines
    //
   
    void GetDiagonal
    ( DistMatrix<T,MD,STAR,Int>& d, Int offset=0 ) const;
    void GetDiagonal
    ( DistMatrix<T,STAR,MD,Int>& d, Int offset=0 ) const;
    void SetDiagonal
    ( const DistMatrix<T,MD,STAR,Int>& d, Int offset=0 );
    void SetDiagonal
    ( const DistMatrix<T,STAR,MD,Int>& d, Int offset=0 );

    // (Immutable) view of a distributed matrix's buffer
    void Attach
    ( Int height, Int width, Int colAlignment, Int rowAlignment,
      T* buffer, Int ldim, const elem::Grid& grid );
    void LockedAttach
    ( Int height, Int width, Int colAlignment, Int rowAlignment,
      const T* buffer, Int ldim, const elem::Grid& grid );      

    // Equate/Update with the scattered summation of A[MC,* ] across process
    // rows
    void SumScatterFrom( const DistMatrix<T,MC,STAR,Int>& A );
    void SumScatterUpdate( T alpha, const DistMatrix<T,MC,STAR,Int>& A );

    // Equate/Update with the scattered summation of A[* ,MR] across process
    // cols
    void SumScatterFrom( const DistMatrix<T,STAR,MR,Int>& A );
    void SumScatterUpdate( T alpha, const DistMatrix<T,STAR,MR,Int>& A );

    // Equate/Update with the scattered summation of A[* ,* ] across the 
    // entire grid.
    void SumScatterFrom( const DistMatrix<T,STAR,STAR,Int>& A );
    void SumScatterUpdate( T alpha, const DistMatrix<T,STAR,STAR,Int>& A );

    // Auxiliary routines needed to implement algorithms that avoid 
    // inefficient unpackings of partial matrix distributions
    void AdjointFrom( const DistMatrix<T,STAR,MC,Int>& A );
    void AdjointFrom( const DistMatrix<T,MR,STAR,Int>& A );
    void AdjointSumScatterFrom( const DistMatrix<T,MR,STAR,Int>& A );
    void AdjointSumScatterUpdate( T alpha, const DistMatrix<T,MR,STAR,Int>& A );

    void TransposeFrom
    ( const DistMatrix<T,STAR,MC,Int>& A, bool conjugate=false );
    void TransposeFrom
    ( const DistMatrix<T,MR,STAR,Int>& A, bool conjugate=false );
    void TransposeSumScatterFrom
    ( const DistMatrix<T,MR,STAR,Int>& A, bool conjugate=false );
    void TransposeSumScatterUpdate
    ( T alpha, const DistMatrix<T,MR,STAR,Int>& A, bool conjugate=false );

    //
    // Though the following routines are meant for complex data, all but two
    // logically apply to real data.
    //

    void GetRealPartOfDiagonal
    ( DistMatrix<typename Base<T>::type,MD,STAR,Int>& d, Int offset=0 ) const;
    void GetImagPartOfDiagonal
    ( DistMatrix<typename Base<T>::type,MD,STAR,Int>& d, Int offset=0 ) const;
    void GetRealPartOfDiagonal
    ( DistMatrix<typename Base<T>::type,STAR,MD,Int>& d, Int offset=0 ) const;
    void GetImagPartOfDiagonal
    ( DistMatrix<typename Base<T>::type,STAR,MD,Int>& d, Int offset=0 ) const;
    void SetRealPartOfDiagonal
    ( const DistMatrix<typename Base<T>::type,MD,STAR,Int>& d, Int offset=0 );
    // Only valid for complex datatypes
    void SetImagPartOfDiagonal
    ( const DistMatrix<typename Base<T>::type,MD,STAR,Int>& d, Int offset=0 );
    void SetRealPartOfDiagonal
    ( const DistMatrix<typename Base<T>::type,STAR,MD,Int>& d, Int offset=0 );
    // Only valid for complex datatypes
    void SetImagPartOfDiagonal
    ( const DistMatrix<typename Base<T>::type,STAR,MD,Int>& d, Int offset=0 );

private:
    virtual void PrintBase( std::ostream& os, const std::string msg="" ) const;

    template<typename S,Distribution U,Distribution V,typename N>
    friend class DistMatrix;
};

} // namespace elem

#endif // ifndef CORE_DISTMATRIX_MC_MR_DECL_HPP