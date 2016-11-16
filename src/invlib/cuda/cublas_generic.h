/**
 * \file cuda/cublas_generic.h
 *
 * \brief Generic wrapper for the CuBLAS library.
 *
 */

#ifndef CUDA_CUBLAS_GENERIC_H
#define CUDA_CUBLAS_GENERIC_H

#include "cuda.h"
#include "cublas_v2.h"

namespace invlib
{
namespace cublas
{

// axpy

template<typename T>
void axpy(cublasHandle_t handle, int m, T alpha,
          const T * x, int x_inc, T * y, int y_inc);

template<>
void axpy<float>(cublasHandle_t handle, int m, float alpha,
                 const float * x, int x_inc, float * y, int y_inc)
{
    cublasSaxpy(handle, m, &alpha, x, x_inc, y, y_inc);
}

template<>
void axpy<double>(cublasHandle_t handle, int m, double alpha,
                  const double * x, int x_inc, double * y, int y_inc)
{
    cublasDaxpy(handle, m, &alpha, x, x_inc, y, y_inc);
}

// dot
template<typename T>
T dot(cublasHandle_t handle, int m, const T * x, int x_inc, const T * y, int y_inc);

template<>
float dot<float>(cublasHandle_t handle, int m,
                 const float * x, int x_inc,
                 const float * y, int y_inc)
{
    float result;
    cublasSdot(handle, m, x, x_inc, y, y_inc, &result);
    return result;
}

template<>
double dot<double>(cublasHandle_t handle, int m,
                   const double * x, int x_inc,
                   const double * y, int y_inc)
{
    double result;
    cublasDdot(handle, m, x, x_inc, y, y_inc, &result);
    return result;
}

}      // namespace cublas
}      // namespace invlib
#endif // CUDA_CUBLAS_GENERIC_H

