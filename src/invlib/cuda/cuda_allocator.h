/**
 * \file cuda/cuda_device.h
 *
 * \brief Memory allocator for cuda devices.
 *
 */

#ifndef CUDA_CUDA_ALLOCATOR_H
#define CUDA_CUDA_ALLOCATOR_H

#include <cassert>
#include <vector>

#include "cuda.h"

#include "invlib/cuda/utility.h"


namespace invlib
{

class CudaAllocator
{
public:

    CudaAllocator() = default;
    CudaAllocator(const CudaAllocator & )              = delete;
    CudaAllocator(      CudaAllocator &&)              = delete;
    CudaAllocator & operator= (const CudaAllocator & ) = delete;
    CudaAllocator & operator= (      CudaAllocator &&) = delete;

    ~CudaAllocator();

    void * request(size_t bytes);
    void   release(void * pointer);

private:

    std::vector<size_t> sizes;
    std::vector<void *> pointers;
    std::vector<bool>   available;

    size_t find_vector(size_t bytes);

};

#include "cuda_allocator.cpp"

}      // namespace invlib
#endif // CUDA_CUDA_ALLOCATOR_H
