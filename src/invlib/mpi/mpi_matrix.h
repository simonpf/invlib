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

    template <typename = enable_if<is_same<StorageType, LocalType>>>
    MPIMatrix();

    MPIMatrix(const MPIMatrix &) = default;
    MPIMatrix(MPIMatrix &&)      = default;

    MPIMatrix & operator=(const MPIMatrix &) = default;
    MPIMatrix & operator=(MPIMatrix &&) = default;

    template <typename T,
              typename = enable_if<is_constructible<StorageType, T>>
              typename = disable_if<is_same<decay<T>, MPIMatrix>>
    MPIMatrix(T &&local_matrix);

    template <typename = enable_if<is_same<StorageType, LocalType>>>
    void resize(unsigned int i, unsigned int j);

    static MPIMatrix<LocalType, LValue> split_matrix(const MatrixType &matrix);
    static void broadcast(LocalType &local);

    unsigned int rows() const;
    unsigned int cols() const;

    template <typename = enable_if<is_same<StorageType, LocalType>>>
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

    StorageType local;
    RealType    local_element;
    unsigned int local_rows;
    unsigned int m, n;

};

#include "mpi_matrix.cpp"

}      // namespace invlib

#endif // MPI_MPI_MATRIX_H

