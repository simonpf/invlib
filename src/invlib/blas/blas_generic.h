/**
 * \file blas/blas_generic.h
 *
 * \brief Generic wrapper for BLAS routines.
 *
 */

#ifndef BLAS_BLAS_GENERIC_H
#define BLAS_BLAS_GENERIC_H

#include "mkl_cblas.h"


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
    return cblas_saxpy(n, alpha, x, incx, y, incy);
}

template<> void axpy(int n,
                       double alpha, const double * x, int incx,
                       double * y, int incy)
{
    return cblas_daxpy(n, alpha, x, incx, y, incy);
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
    return cblas_sdot(n, x, incx,  y, incy);

}

template<> double dot<double>(int n, const double * x, int incx,
                                     const double * y, int incy)
{
    return cblas_ddot(n, x, incx, y, incy);
}

}      // namespace blas
}      // namespace invlib
#endif // BLAS_BLAS_GENERIC_H
