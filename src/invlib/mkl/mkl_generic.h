/**
 * \file blas/blas_generic.h
 *
 * \brief Generic wrapper for sparse MKL routines.
 *
 */

#ifndef MKL_MKL_GENERIC_H
#define MKL_MKL_GENERIC_H

#include "mkl.h"

namespace invlib {
namespace mkl {

matrix_descr matrix_descriptor = {SPARSE_MATRIX_TYPE_GENERAL,
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

//--------------------------------//
//  Matrix Vector Multiplication  //
//--------------------------------//


template<typename Real, Representation rep>
    void mv(const sparse_operation_t operation,
            const Real alpha,
            const sparse_matrix_t A,
            const Real * x,
            const Real beta,
                  Real * y);

template<>
void mv<float, Representation::CompressedRows>(
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
    void mv<double, Representation::CompressedRows>(
        const sparse_operation_t operation,
        const double alpha,
        const sparse_matrix_t A,
        const double * x,
        const double beta,
              double * y)
{
    mkl_sparse_d_mv(operation, alpha, A, matrix_descriptor, x, beta, y);
}

const char mkl_matrix_descriptor[4] = {'G',' ',' ', 'C'};

template<typename Real, Representation rep> void smv(
    char transa,
    int  m, int k, int nnz,
    Real alpha,
    const Real *, const int *, const int *, const int *,
    const Real * x,
    Real beta, Real * y);

template<> void smv<float, Representation::Coordinates>(
    char transa,
    int m, int k, int nnz,
    float alpha,
    const float * a, const int * row_indices, const int * column_indices,
    const int * /*not used*/,
    const float * x,
    float beta, float * y)
{
    mkl_scoomv(&transa, &m, &k, &alpha, mkl_matrix_descriptor,
               a, row_indices, column_indices, &nnz, x, &beta, y);
}

template<> void smv<double, Representation::Coordinates>(
    char transa,
    int m, int k, int nnz,
    double alpha,
    const double * a, const int * row_indices, const int * column_indices,
    const int * /*not used*/,
    const double * x,
    double beta, double * y)
{
    mkl_dcoomv(&transa, &m, &k, &alpha, mkl_matrix_descriptor,
               a, row_indices, column_indices, &nnz, x, &beta, y);
}

template<> void smv<float, Representation::CompressedColumns>(
    char transa,
    int m, int k, int nnz,
    float alpha,
    const float * a, const int * row_indices,
    const int * column_starts, const int * column_ends,
    const float * x,
    float beta, float * y)
{
    mkl_scscmv(&transa, &m, &k, &alpha, mkl_matrix_descriptor,
               a, row_indices, column_starts, column_ends, x, &beta, y);
}

template<> void smv<double, Representation::CompressedColumns>(
    char transa,
    int m, int k, int nnz,
    double alpha,
    const double * a, const int * row_indices,
    const int * column_starts, const int * column_ends,
    const double * x,
    double beta, double * y)
{
    mkl_dcscmv(&transa, &m, &k, &alpha, mkl_matrix_descriptor,
               a, row_indices, column_starts, column_ends, x, &beta, y);
}

template<> void smv<float, Representation::CompressedRows>(
    char transa,
    int m, int k, int nnz,
    float alpha,
    const float * a, const int * column_indices,
    const int * row_starts, const int * row_ends,
    const float * x,
    float beta, float * y)
{
    mkl_scsrmv(&transa, &m, &k, &alpha, mkl_matrix_descriptor,
               a, column_indices, row_starts, row_ends, x, &beta, y);
}

template<> void smv<double, Representation::CompressedRows>(
    char transa,
    int m, int k, int nnz,
    double alpha,
    const double * a, const int * column_indices,
    const int * row_starts, const int * row_ends,
    const double * x,
    double beta, double * y)
{
    mkl_dcsrmv(&transa, &m, &k, &alpha, mkl_matrix_descriptor,
               a, column_indices, row_starts, row_ends, x, &beta, y);
}

}      // namespace invlib
}      // namespace mkl
#endif // MKL_MKL_GENERIC_H
