/**
 * \file cuda/cuda_device.h
 *
 * \brief Class representing cuda devices.
 *
 */

#ifndef CUDA_CUDA_DEVICE_H
#define CUDA_CUDA_DEVICE_H

#include "cuda.h"
#include "cublas_v2.h"
#include "cuda_allocator.h"

namespace invlib
{

class CudaDevice
{
public:

    CudaDevice();
    CudaDevice(const CudaDevice & )              = delete;
    CudaDevice(      CudaDevice &&)              = delete;
    CudaDevice & operator= (const CudaDevice & ) = delete;
    CudaDevice & operator= (      CudaDevice &&) = delete;

    CudaAllocator & get_allocator()           {return cuda_allocator;}
    cublasHandle_t  get_cublas_handle() const {return cublas_handle;}

    dim3 get_1d_block()     const;
    dim3 get_1d_grid(int m) const;

private:

    CudaAllocator  cuda_allocator;
    cublasHandle_t cublas_handle;

    int block_size = 256;
};

struct Cuda
{
    static CudaDevice default_device;
};

#include "cuda_device.cpp"

}      // namespace invlib
#endif // CUDA_CUDA_DEVICE_H
