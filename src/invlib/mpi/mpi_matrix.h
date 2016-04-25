/** \file mpi/mpi_matrix.h
 *
 * \brief Contains the MPIMatrix class, a generic class for matrices distributed
 * row-wise over nodes.
 *
 */

#ifndef MPI_MPI_MATRIX_H
#define MPI_MPI_MATRIX_H

#include <utility>
#include <vector>
#include "mpi.h"
#include "invlib/traits.h"
#include "invlib/mpi/traits.h"

namespace invlib
{

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
 * In addition the associated VectorType must provide a raw_pointer() function
 * so that the results can be broad casted using MPI.
 *
 * The MPI matrix class holds matrix block local to the process as lvalue or
 * as reference.
 *
 * \tparam LocalType The type of the local matrix block.
 * \tparam StorageTrait Storage template that defines whether the local block
 * is held as a reference or as lvalue. See invlib/mpi/traits.h.
 *
 */
template
<
typename LocalType,
template <typename> typename StorageTrait = ConstRef
>
class MPIMatrix
{

public:

    // -------------- //
    //  Type Aliases  //
    // -------------- //

    /*! The basic scalar type. */
    using RealType   = typename LocalType::RealType;
    /*! The basic vector type  */
    using VectorType = typename LocalType::VectorType;
    /*! The local Matrix type.  */
    using MatrixType = LocalType;
    /*!
     * Result type of an algebraic expression with MPIMatrix as right hand
     * operator.
     */
    using ResultType = LocalType;
    /*! The type used to store the local matrix. */
    using StorageType = typename StorageTrait<LocalType>::type;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    MPIMatrix();

    MPIMatrix(const MPIMatrix &) = default;
    MPIMatrix(MPIMatrix &&)      = default;

    MPIMatrix & operator=(const MPIMatrix &) = default;
    MPIMatrix & operator=(MPIMatrix &&) = default;

    template <typename T,
              typename = enable_if<is_constructible<CopyWrapper<StorageType>, T>>,
              typename = disable_if<is_same<decay<T>, MPIMatrix>>>
    MPIMatrix(T &&local_matrix);

    MPIMatrix(const LocalType &local_matrix);

    void resize(unsigned int i, unsigned int j);


    static void broadcast(LocalType &local);
    MPIMatrix<LocalType, LValue> split_matrix(const MatrixType &local_matrix);

    unsigned int rows() const;
    unsigned int cols() const;

    LocalType& get_local();

    RealType operator()(unsigned int i, unsigned int j) const;
    RealType& operator()(unsigned int i, unsigned int j);

    VectorType multiply(const VectorType &) const;
    VectorType transpose_multiply(const VectorType &) const;

    /* operator MPIMatrix<LocalType, ConstRef>() const; */
    /* operator LocalType(); */

private:

    void broadcast_local_rows(int proc_rows[]) const;
    void broadcast_local_block(double *vector,
                                const double *block) const;
    void reduce_vector_sum(double *result_vector,
                           double *local_vector) const;

    static constexpr MPI_Datatype mpi_data_type = MPI_DOUBLE;
    std::vector<unsigned int> row_indices;
    std::vector<unsigned int> row_ranges;

    int rank;
    int nprocs;

    CopyWrapper<StorageType> local;
    RealType    local_element;
    unsigned int local_rows;
    unsigned int m, n;

};

#include "mpi_matrix.cpp"

}      // namespace invlib

#endif // MPI_MPI_MATRIX_H

