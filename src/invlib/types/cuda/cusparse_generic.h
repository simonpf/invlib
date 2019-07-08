/**
 * \file cuda/cusparse_generic.h
 *
 * \brief Generic wrappers for the cuSPARSE library.
 *
 */

#ifndef CUDA_CUSPARSE_GENERIC_H
#define CUDA_CUSPARSE_GENERIC_H

#include "cuda.h"
#include "cusparse_v2.h"

#include "invlib/sparse/sparse_data.h"

namespace invlib
{
namespace cusparse
{

inline cusparseOperation_t invert(cusparseOperation_t trans)
{
    if (trans == CUSPARSE_OPERATION_TRANSPOSE)
    {
        return CUSPARSE_OPERATION_NON_TRANSPOSE;
    }
    else
    {
        return CUSPARSE_OPERATION_TRANSPOSE;
    }
}

template<typename Real, Representation rep> void smv(
    cusparseHandle_t, cusparseOperation_t, int, int, int, Real,
    const cusparseMatDescr_t,
    const Real *, const int  *, const int *,
    const Real *, Real,
    Real *);

template<>
void smv<float, Representation::CompressedColumns>(
    cusparseHandle_t handle, cusparseOperation_t transA,
    int m, int n, int nnz, float alpha,
    const cusparseMatDescr_t desc,
    const float * cscValA,
    const int  * cscColPtrA, const int * cscRowIndA,
    const float  * x, float beta,
    float * y)
{
    cusparseScsrmv(handle, invert(transA), n, m, nnz, &alpha, desc,
                   cscValA, cscColPtrA, cscRowIndA,
                   x, &beta, y);
}

template<>
void smv<double, Representation::CompressedColumns>(
    cusparseHandle_t handle, cusparseOperation_t transA,
    int m, int n, int nnz, double alpha,
    const cusparseMatDescr_t desc,
    const double * cscValA,
    const int  * cscColPtrA, const int * cscRowIndA,
    const double  * x, double beta,
    double * y)
{
    cusparseStatus_t e = cusparseDcsrmv(
        handle, invert(transA), n, m, nnz, &alpha, desc,
        cscValA, cscColPtrA, cscRowIndA,
        x, &beta, y);
}

template<>
void smv<float, Representation::CompressedRows>(
    cusparseHandle_t handle, cusparseOperation_t transA,
    int m, int n, int nnz, float alpha,
    const cusparseMatDescr_t desc,
    const float * csrValA,
    const int  * csrRowPtrA, const int * csrColIndA,
    const float  * x, float beta,
    float * y)
{
    cusparseScsrmv(handle, transA, m, n, nnz, &alpha, desc,
                   csrValA, csrRowPtrA, csrColIndA,
                   x, &beta, y);
}

template<>
void smv<double, Representation::CompressedRows>(
    cusparseHandle_t handle, cusparseOperation_t transA,
    int m, int n, int nnz, double alpha,
    const cusparseMatDescr_t desc,
    const double * csrValA,
    const int  * csrRowPtrA, const int * csrColIndA,
    const double  * x, double beta,
    double * y)
{
    cusparseStatus_t e = cusparseDcsrmv(
        handle, transA, m, n, nnz, &alpha, desc,
        csrValA, csrRowPtrA, csrColIndA,
        x, &beta, y);
}

}      // namespace cusparse
}      // namespace invlib
#endif // CUDA_CUSPARSE_GENERIC_H
