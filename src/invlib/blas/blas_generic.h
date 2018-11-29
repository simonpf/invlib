/**
 * \file blas/blas_generic.h
 *
 * \brief Generic wrapper for BLAS routines.
 *
 */

#ifndef BLAS_BLAS_GENERIC_H
#define BLAS_BLAS_GENERIC_H

extern "C" void saxpy_(int *n, float *alpha, const float* x, int *inc_x,
                      float *y, int *inc_y);
extern "C" void daxpy_(int *n, double *alpha, const double* x, int *inc_x,
                      double *y, int *inc_y);

extern "C" float  sdot_(int *n, const float  *x,  int *inc_x, const float  *y, int *inc_y);
extern "C" double ddot_(int *n, const double *x,  int *inc_x, const double *y, int *inc_y);

extern "C" void sgemm_(char * transa, char * transb, int * m, int * n, int * k,
                       float * alpha, const float * A, int * lda, const float * B,
                       int * ldb, float * beta, float * C, int * ldc);
extern "C" void dgemm_(char * transa, char * transb, int * m, int  * n, int * k,
                       double * alpha, const double * A, int * lda, const double * B,
                       int * ldb, double * beta, double * C, int * ldc);

extern "C" void sgemv_(char * trans, int * m, int * n, float * alpha, const float * A,
                       int  * lda, const float * x, int * incx, float * beta, float * y,
                       int * incy);
extern "C" void dgemv_(char * trans, int * m, int * n, double * alpha, const double * A,
                       int  * lda, const double * x, int * incx, double * beta, double * y,
                       int * incy);

namespace invlib {
namespace blas {

/*! Generic cBLAS axpy
 *
 * Wrapper for the cBLAS saxpy and daxpy functions that perform scaled
 * addition of two vectors:
 * \f[
 *  \vec{y} = \vec{y} + \alpha \vec{x}
 * \f]
 *
 * For a detailed description of the functions arguments see cBLAS documentation.
 *
 * \tparam Floating point type used for the representation of vector elements.
 * Supported types: float, double.
 */
template<typename Real> void axpy(int n,
                                  Real alpha, const Real * x, int incx,
                                  Real * y, int incy);

template<> void axpy(int n,
                     float alpha, const float * x, int incx,
                     float * y, int incy)
{
    return saxpy_(&n, &alpha, x, &incx, y, &incy);
}

template<> void axpy(int n,
                     double alpha, const double * x, int incx,
                     double * y, int incy)
{
    return daxpy_(&n, &alpha, x, &incx, y, &incy);
}

/*! Generic gemm
 *
 * Wrapper for the BLAS sgemm and dgemm functions for matrix
 * multiplication.
 *
 * \tparam Floating point type used for the representation of vector elements.
 * Supported types: float, double.
 */
template<typename Real> void gemm(char transa, char transb,
                                  int m, int n, int k,
                                  Real alpha, const Real * A, int lda,
                                  const Real * B, int ldb, Real beta,
                                  Real * C, int ldc);

template<> void gemm(char transa, char transb,
                     int m, int n, int k,
                     float alpha, const float * A, int lda,
                     const float * B, int ldb, float beta,
                     float * C, int ldc) {
    return sgemm_(&transa, &transb, &m, &n, &k,
                  &alpha, A, &lda,
                  B, &ldb, &beta,
                  C, &ldc);
}

template<> void gemm(char transa, char transb,
                     int m, int n, int k,
                     double alpha, const double * A, int lda,
                     const double * B, int ldb, double beta,
                     double * C, int ldc) {
    return dgemm_(&transa, &transb, &m, &n, &k,
                  &alpha, A, &lda,
                  B, &ldb, &beta,
                  C, &ldc);
}

/*! Generic gemv
 *
 * Wrapper for the BLAS sgemm and dgemm functions for matrix-vector
 * multiplication.
 *
 * \tparam Floating point type used for the representation of vector elements.
 * Supported types: float, double.
 */
template<typename Real> void gemv(char trans, int m, int n,
                                  Real alpha, const Real * A, int lda,
                                  const Real * x, int incx,
                                  Real beta, Real * y, int incy);

template<> void gemv(char trans, int m, int n,
                     float alpha, const float * A, int lda,
                     const float * x, int incx,
                     float beta, float * y, int incy)
{
    sgemv_(&trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

template<> void gemv(char trans, int m, int n,
                     double alpha, const double * A, int lda,
                     const double * x, int incx,
                     double beta, double * y, int incy)
{
    dgemv_(&trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

/*! Generic cBLAS dot
 *
 * Wrapper for the cBLAS sdot and ddot functions that compute the
 * dot product of two vectors:
 * \f[
 *  \sum_{i = 0}^n x_i y_i
 * \f]
 *
 * For a detailed description of the functions arguments see cBLAS documentation.
 *
 * \tparam Floating point type used for the representation of vector elements.
 * Supported types: float, double.
 */
template<typename Real> Real dot(int n, const Real * x, int incx,
                                        const Real * y, int incy);

template<> float dot<float>(int n, const float * x, int incx,
                                   const float * y, int incy)
{
    return sdot_(&n, x, &incx,  y, &incy);
}

template<> double dot<double>(int n, const double * x, int incx,
                                     const double * y, int incy)
{
    return ddot_(&n, x, &incx, y, &incy);
}

}      // namespace blas
}      // namespace invlib
#endif // BLAS_BLAS_GENERIC_H
