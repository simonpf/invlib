/**
 * \file cuda/cuda_vector.h
 *
 * \brief Utility classes and functions for CUDA matrix and vector
 * types.
 *
 */

#ifndef CUDA_UTILITY_H
#define CUDA_UTILITY_H

#include "stdio.h"
#include "cuda_runtime_api.h"

namespace invlib
{

void handle_cuda_error(cudaError_t code,
                       const char *file,
                       int line,
                       bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
   }
}

template <typename T>
struct CudaDeleter
{
    CudaDeleter()                     = default;
    CudaDeleter(const CudaDeleter & ) = default;
    CudaDeleter(      CudaDeleter &&) = default;

    void operator() (const T * ptr)
    {
        if (* ptr) {
            cudaFree(static_cast<void **>(ptr));
        }
        delete ptr;
    }
};

}      // namespace invlib
#endif // CUDA_UTILITY_H
