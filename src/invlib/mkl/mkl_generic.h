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
