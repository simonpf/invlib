/** \file mpi/mpi_matrix.h
 *
 * \brief Contains the MpiMatrix class, a generic class for matrices distributed
 * row-wise over nodes.
 *
 */

#ifndef MPI_MPI_MATRIX_H
#define MPI_MPI_MATRIX_H

#include <utility>
#include <vector>
#include <iostream>
#include "mpi.h"
#include "invlib/traits.h"
#include "invlib/mpi/generic.h"
#include "invlib/mpi/traits.h"

namespace invlib
{

template
<
typename Base
>
class Vector;

template
<
typename LocalType,
template <typename> class StorageTrait
>
class MpiVector;

// -------------- //
//  Matrix Class  //
// -------------- //

/**
 * \brief Generic distributed MPI matrix.
 *
 * Provides a wrapper class for matrices that are distributed row-wise
 * over processes. Currently only matrix-vector multiplication and
 * transposed vector-matrix multiplication are supported. This requires
 * the underlying matrix type to implement the following operations:
 *
 * - MV multiplicaiton: multiply(const VectorType &v)
 * - transposed MV multiplication by a block:
 *     transpose_multiply_block(const VectorType &v, int start, int extent)
 *
 * In addition the associated VectorType must provide a data_pointer() function
 * so that the results can be broad casted using MPI.
 *
 * The MPI matrix class holds the local matrix block of the process as
 * lvalue or as reference. We refer to the type of the matrix used for
 * each local block as the local type. The way in wich the local type
 * is stored is refered to as the storage type . Currently two storage
 * types are supported: Storing the local matrix as a reference
 * (default) and storing the lo local matrix as an lvalue,
 * i.e. copying / moving it into the MpiMatrix object on construction.
 *
 * \attention When storing the local matrix by reference implicit conversion
 * may lead to dangling references.
 *
 * \tparam LocalType The type of the local matrix block.
 * \tparam StorageTrait Storage template that defines whether the local block
 * is held as a reference or as lvalue. See invlib/mpi/traits.h.
 *
 */
template
<
typename LocalType,
template <typename> class StorageTrait = ConstRef
>
class MpiMatrix
{

public:

    // -------------- //
    //  Type Aliases  //
    // -------------- //

    /*! The basic scalar type. */
    using RealType   = typename LocalType::RealType;
    /*! The MPI type corresponding to the local vector type. */
    using NonMpiVectorType = typename LocalType::VectorType;
    /*! The local Matrix type.  */
    template<template <typename> class VectorStorageType>
    using MpiVectorType = Vector<MpiVector<typename LocalType::VectorType,
                                           VectorStorageType>>;
    /*! The basic vector type  */
    using VectorType = typename LocalType::VectorType;
    /*! The basic vector type  */
    using ResultType = typename LocalType::VectorType;
    /*! The local Matrix type.  */
    using MatrixType = LocalType;
    /*! The type used to store the local matrix. */
    using StorageType = typename StorageTrait<LocalType>::type;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    /*! Default Constructor.
     *
     * Works only if the local matrix is stored as lvalue.
     */
    MpiMatrix();

    MpiMatrix(const MpiMatrix &) = default;
    MpiMatrix(MpiMatrix &&)      = default;

    MpiMatrix & operator=(const MpiMatrix &);
    MpiMatrix & operator=(MpiMatrix &&) = default;

    /*!
     * Generic constructor that forwards the call to the constructor of
     * local type so that the MPI matrix can be constructed from any type
     * the local type can be constructed from.
     */
    template <typename T,
              typename = enable_if<is_constructible<CopyWrapper<StorageType>, T>>,
              typename = disable_if<is_same<decay<T>, MpiMatrix>>>
    MpiMatrix(T &&local_matrix);

    template <typename T, typename U, typename ...Vs>
    MpiMatrix(T &&t, U && u, Vs && ... vs) :
    local(std::forward<T>(t), std::forward<U>(u),  std::forward<Vs>(vs) ...) {
    // Nothing to do here.
    }


    /*!
     * Construct MPI matrix from local matrix. The constructor assumes that
     * every MPI process holds a local block of a matrix with the rows
     * distributed contiguously and in increasing order (but not
     * necessarily uniformly) over MPI ranks. The constructed MpiMatrix object
     * represents the full matrix with m rows, where m is the sum of all rows
     * of the matrices in each MPI process.
     *
     * \param local_matrix The local block of each process represented by
     * a matrix of the given local type.
     */

    MpiMatrix(const LocalType &local_matrix);

    // --------------- //
    //  Manipulations  //
    // --------------- //

    /*!
     * Resize the local matrix. Resizes the local matrix held by each MPI
     * process. This method is only available if the local matrix is mutable,
     * i.e. is stored as lvaule. The resulting MpiMatrix object represents
     * the global \f$n_{\text{proc}} * i \times j \f$ matrix where
     *  \f$n_\text{proc}\f$ is the number of MPI processes.
     *
     * \param i The number of local rows.
     * \param j The number of columns
     */
    void resize(unsigned int i, unsigned int j);


    /** Broadcast the local matrix of process 0 to all other
     *  processes. After the call  to broadcast all processes will have
     *  the same local matrix as process 0. */
    static void distribute(LocalType &local);

    /** Split the matrix \p local_matrix evenly over processes. Splits up the rows
     * of the matrix \p local_matrix evenly over the running MPI processes and
     * creates an MPI matrix with lvalue storage type by copying the block from
     * \p local_matrix that corresponds to the row-range of each process. The
     * returned MPI matrix is a distributed representation of \p local_matrix. */
    static MpiMatrix<LocalType, LValue> split_matrix(const MatrixType &local_matrix);

    size_t rows() const;
    size_t cols() const;

    MpiVectorType<LValue> col(size_t i) const;
    NonMpiVectorType      row(size_t i) const;

    MpiVectorType<LValue> diagonal() const;

    LocalType& get_local();

    RealType operator()(unsigned int i, unsigned int j) const;
    RealType& operator()(unsigned int i, unsigned int j);

    template <typename T>
    auto multiply(T &) const -> typename T::ResultType;

    template <typename T>
    auto transpose_multiply(T &) const -> typename T::ResultType;

    template <template <typename> class VectorStorageType>
    auto multiply(const MpiVectorType<VectorStorageType> &v) const
        -> MpiVectorType<LValue>;

    template <template <typename> class VectorStorageType>
    auto transpose_multiply(const MpiVectorType<VectorStorageType> &v) const
        -> MpiVectorType<LValue>;

protected:

    static constexpr MPI_Datatype mpi_data_type = MpiDataType<RealType>::name;

    void broadcast_local_rows(int proc_rows[]) const;
    void broadcast_local_block(RealType *vector,
                                const RealType *block) const;
    void reduce_vector_sum(RealType *result_vector,
                           RealType *local_vector) const;

    std::vector<unsigned int> row_indices;
    std::vector<unsigned int> row_ranges;

    int rank;
    int nprocs;

    CopyWrapper<StorageType> local;
    RealType     local_element;
    unsigned int local_rows;
    unsigned int m, n;

};

#include "mpi_matrix.cpp"

}      // namespace invlib
#endif // MPI_MPI_MATRIX_H

