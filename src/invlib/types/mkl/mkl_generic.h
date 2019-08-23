/**
 * \file blas/blas_generic.h
 *
 * \brief Generic wrapper for sparse MKL routines.
 *
 */

#ifndef MKL_MKL_GENERIC_H
#define MKL_MKL_GENERIC_H

#include <invlib/types/sparse/sparse_data.h>

#if defined MKL_AVAILABLE
#include "mkl.h"
#include "mkl_spblas.h"

namespace invlib {
namespace mkl {

::matrix_descr matrix_descriptor = {SPARSE_MATRIX_TYPE_GENERAL,
                                    SPARSE_FILL_MODE_FULL,
                                    SPARSE_DIAG_NON_UNIT};

sparse_index_base_t index_base = SPARSE_INDEX_BASE_ZERO;

// --------------- //
//  Sparse Create  //
// --------------- //

template<typename Real, Representation rep>
sparse_matrix_t sparse_create(MKL_INT, MKL_INT, MKL_INT *, MKL_INT *, Real *);

template<>
sparse_matrix_t sparse_create<float, Representation::CompressedRows>(
    MKL_INT m,
    MKL_INT n,
    MKL_INT * rows_start,
    MKL_INT * col_indx,
    float * values)
{
    sparse_matrix_t A;
    sparse_status_t s = mkl_sparse_s_create_csr(&A,
                                                index_base,
                                                m, n,
                                                rows_start, rows_start + 1,
                                                col_indx,
                                                values);

    if (!(s == SPARSE_STATUS_SUCCESS)) {
        throw std::runtime_error("Error constructing sparse MKL matrix.");
    }

    return A;
}

template<>
sparse_matrix_t sparse_create<double, Representation::CompressedRows>(
    MKL_INT m,
    MKL_INT n,
    MKL_INT * rows_start,
    MKL_INT * col_indx,
    double * values)
{
    sparse_matrix_t A;
    sparse_status_t s = mkl_sparse_d_create_csr(&A,
                                                index_base,
                                                m, n,
                                                rows_start, rows_start + 1,
                                                col_indx,
                                                values);

    if (!(s == SPARSE_STATUS_SUCCESS)) {
        throw std::runtime_error("Error constructing sparse MKL matrix.");
            }

    return A;
}

template<>
sparse_matrix_t sparse_create<float, Representation::CompressedColumns>(
    MKL_INT m,
    MKL_INT n,
    MKL_INT * cols_start,
    MKL_INT * row_indx,
    float * values)
{
    sparse_matrix_t A;
    sparse_status_t s = mkl_sparse_s_create_csc(&A,
                                                index_base,
                                                m, n,
                                                cols_start, cols_start + 1,
                                                row_indx,
                                                values);

    if (!(s == SPARSE_STATUS_SUCCESS)) {
        throw std::runtime_error("Error constructing sparse MKL matrix.");
    }

    return A;
}

template<>
    sparse_matrix_t sparse_create<double, Representation::CompressedColumns>(
        MKL_INT m,
        MKL_INT n,
        MKL_INT * cols_start,
        MKL_INT * row_indx,
        double * values)
{
    sparse_matrix_t A;
    sparse_status_t s = mkl_sparse_d_create_csc(&A,
                                                index_base,
                                                m, n,
                                                cols_start, cols_start + 1,
                                                row_indx,
                                                values);

    if (!(s == SPARSE_STATUS_SUCCESS)) {
        throw std::runtime_error("Error constructing sparse MKL matrix.");
    }

    return A;
}

template<>
sparse_matrix_t sparse_create<float, Representation::Coordinates>(
    MKL_INT m,
    MKL_INT n,
    MKL_INT * cols_start,
    MKL_INT * row_indx,
    float * values)
{
    sparse_matrix_t A;
    sparse_status_t s = mkl_sparse_s_create_csc(&A,
                                                index_base,
                                                m, n,
                                                cols_start, cols_start + 1,
                                                row_indx,
                                                values);

    if (!(s == SPARSE_STATUS_SUCCESS)) {
        throw std::runtime_error("Error constructing sparse MKL matrix.");
    }

    return A;
}

sparse_matrix_t sparse_create_coo(MKL_INT m,
                                  MKL_INT n,
                                  MKL_INT nnz,
                                  MKL_INT * row_indices,
                                  MKL_INT * column_indices,
                                  double * values)
{
    sparse_matrix_t A;
    sparse_status_t s = mkl_sparse_d_create_coo(&A, index_base, m, n, nnz,
                                                row_indices, column_indices, values);

    if (!(s == SPARSE_STATUS_SUCCESS)) {
        throw std::runtime_error("Error constructing sparse MKL matrix.");
    }

    return A;
}

sparse_matrix_t sparse_create_coo(MKL_INT m,
                                  MKL_INT n,
                                  MKL_INT nnz,
                                  MKL_INT * row_indices,
                                  MKL_INT * column_indices,
                                  float * values)
{
    sparse_matrix_t A;
    sparse_status_t s = mkl_sparse_s_create_coo(&A, index_base, m, n, nnz,
                                                row_indices, column_indices, values);

    if (!(s == SPARSE_STATUS_SUCCESS)) {
        throw std::runtime_error("Error constructing sparse MKL matrix.");
    }

    return A;
}

//--------------------------------//
//  Matrix Vector Multiplication  //
//--------------------------------//

template<typename Real>
    void mv(const sparse_operation_t operation,
            const Real alpha,
            const sparse_matrix_t A,
            const Real * x,
            const Real beta,
                  Real * y);

template<>
void mv<float>(
    const sparse_operation_t operation,
    const float alpha,
    const sparse_matrix_t A,
    const float * x,
    const float beta,
          float * y)
{
    mkl_sparse_s_mv(operation, alpha, A, matrix_descriptor, x, beta, y);
}

template<>
void mv<double>(
        const sparse_operation_t operation,
        const double alpha,
        const sparse_matrix_t A,
        const double * x,
        const double beta,
              double * y)
{
    mkl_sparse_d_mv(operation, alpha, A, matrix_descriptor, x, beta, y);
}

}      // namespace invlib
}      // namespace mkl
#else  // MKL_AVAILABLE

using MKL_INT = size_t;
using sparse_matrix_t = size_t;
using sparse_operation_t = size_t;
const size_t SPARSE_OPERATION_TRANSPOSE = 0;
size_t SPARSE_OPERATION_NON_TRANSPOSE = 0;

namespace invlib {
namespace mkl {

template<typename Real, Representation rep>
sparse_matrix_t sparse_create(MKL_INT, MKL_INT, MKL_INT *, MKL_INT *, Real *);
sparse_matrix_t sparse_create_coo(MKL_INT, MKL_INT, MKL_INT, MKL_INT *, MKL_INT *, double *);
template<typename T> void mv(const sparse_operation_t, const T, const sparse_matrix_t,
                             const T * x, const T, T *);
}
}

#endif // MKL_AVAILABLE
#endif // MKL_MKL_GENERIC_H
