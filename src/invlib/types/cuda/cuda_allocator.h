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

/** Custom Cuda Allocator
 *
 * Manages a pool of memory allocation on a CUDA device. After creation of the
 * object, the pool is empty. If a given number of bytes is requested using
 * the request() member function, the memory is allocated using cudaMalloc(...).
 * When the memory is released the memory is not freed on the device but added to
 * the memory pool and assigned to the next request with as many or less bytes.
 *
 * The device memory is freed on destruction of the allocator object.
 */
class CudaAllocator
{
public:

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    CudaAllocator() = default;
    CudaAllocator(const CudaAllocator & )              = delete;
    CudaAllocator(      CudaAllocator &&)              = delete;
    CudaAllocator & operator= (const CudaAllocator & ) = delete;
    CudaAllocator & operator= (      CudaAllocator &&) = delete;

    ~CudaAllocator();

    // ----------------------------------- //
    //  Memory Allocation and Deallocation //
    // ----------------------------------- //

    /*! Request device memory.
     *
     * Searches through the memory pool to check whether a previously
     * allocated and released array matches the requested size. If
     * such an array is available, the corresponding pointer is
     * returned. If not, the memory is allocated using cudaMalloc(...)
     * and the corresponding pointer is returned.
     */
    void * request(size_t bytes);
    /*! Release device memory.
     *
     * The argument provided to the functions must be a pointer to an array
     * on the device that has previously been allocated using the request(...)
     * member functions of the same allocator. The release function adds the
     * array to the memory pool, so that the next call to request(...) with
     * less or equally as many bytes can return the pointer to the released array.
     */
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
