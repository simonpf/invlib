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

/*! CudaDevice class representing a CUDA compute device.
 *
 * The CudaDevice class holds device specific ressources and settings.
 * Each CUDA vector or matrix holds a reference to a CudaDevice object,
 * which represents the device that the object is stored on.
 *
 * Each CudaDevice object holds a CuBLAS and CuSparse handle as well as
 * a CudaAllocator object.
 */
class CudaDevice
{
public:

    CudaDevice();
    CudaDevice(const CudaDevice & )              = delete;
    CudaDevice(      CudaDevice &&)              = delete;
    CudaDevice & operator= (const CudaDevice & ) = delete;
    CudaDevice & operator= (      CudaDevice &&) = delete;

    CudaAllocator &       get_allocator()             {return cuda_allocator;}
    cublasHandle_t        get_cublas_handle()   const {return cublas_handle;}
    cusparseHandle_t      get_cusparse_handle() const {return cusparse_handle;}

    dim3 get_1d_block()     const;
    dim3 get_1d_grid(int m) const;

private:

    CudaAllocator      cuda_allocator;
    cublasHandle_t     cublas_handle;
    cusparseHandle_t   cusparse_handle;

    int block_size = 256;
};

/*! Struct holding the CUDA default device. */
struct Cuda
{
    static CudaDevice default_device;
};

#include "cuda_device.cpp"

}      // namespace invlib
#endif // CUDA_CUDA_DEVICE_H
