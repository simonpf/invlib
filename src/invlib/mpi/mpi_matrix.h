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

namespace invlib
{

template
<
typename T
>
struct ConstRef
{
    using type = const T &;
};

template
<
typename T
>
struct LValue
{
    using type = T;
};

template
<
typename LocalType,
template <typename> typename StorageTrait = ConstRef
>
class MPIMatrix
{

public:

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

    template
    <
    typename T,
    typename = enable_if<is_constructible<StorageType, T>>
    >
    MPIMatrix(T &&local_matrix);

    static MPIMatrix<LocalType, LValue> split_matrix(const MatrixType &matrix);
    static void broadcast(LocalType &local);

    unsigned int rows() const;
    unsigned int cols() const;

    VectorType multiply(const VectorType &) const;
    VectorType transpose_multiply(const VectorType &) const;

private:

    void broadcast_local_rows(int proc_rows[]) const;
    void broadcast_local_blocks(double *vector,
                                const double *block) const;
    void reduce_vector_sum(double *result_vector,
                           double *local_vector) const;

    static constexpr MPI_Datatype mpi_data_type = MPI_DOUBLE;
    std::vector<unsigned int> row_indices;
    std::vector<unsigned int> row_ranges;

    int rank;
    int nprocs;

    StorageType local;
    unsigned int local_rows;
    unsigned int m, n;

};

#include "mpi_matrix.cpp"

}      // namespace invlib

#endif // MPI_MPI_MATRIX_H

