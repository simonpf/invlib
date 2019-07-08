/**
 * \file cuda/cuda_vector.h
 *
 * \brief Dense vector class for cuda architectures.
 *
 */
#ifndef CUDA_CUDA_VECTOR_H
#define CUDA_CUDA_VECTOR_H

#include <memory>

#include "cuda_runtime_api.h"
#include "invlib/dense/vector_data.h"
#include "invlib/cuda/cublas_generic.h"
#include "invlib/cuda/cuda_device.h"
#include "invlib/cuda/kernels.h"
#include "invlib/cuda/utility.h"

namespace invlib {

// -------------------- //
// Forward Declarations //
// -------------------- //

template <typename RealType> class CudaVector;
template <typename Real> class CudaMatrix
{
public:
    using RealType   = Real;
    using VectorType = CudaVector<RealType>;
    using MatrixType = CudaMatrix<RealType>;
    using ResultType = CudaVector<RealType>;
};
template <typename RealType, Representation rep>  class CudaSparse;

template <typename RealType>
RealType dot(const CudaVector<RealType>&, const CudaVector<RealType>&);

// -------------------  //
//   Cuda Vector Class  //
// -------------------  //

/**
 * \brief Dense Vector Class for CUDA Architectures
 *
 * The CudaVector class template represents dense vectors of with single or
 * double precision elements for computations on Cuda devices.
 *
 * The device memory is handled using a shared_ptr with custom deleter. Copying
 * a cuda vector thus results in a shallow copy, meaning that manipulations on
 * a copied CudaVector object will also affect the original vector.
 *
 * The device memory to hold the vector elements is managed using a CudaAllocator
 * object, which manages a memory pool of allocated arrays on the device. The
 * aim of this indirection is to avoid frequent reallocation of device memory
 * for vectors holding intermediate results.
 *
 * Arithmetic operations (except for addition of a scalar and scaling by a scalar)
 * are implemented using CuBLAS.
 *
 * \tparam Real Floating point type used for the representation of
 * the vector elements.
 */
template
<
typename Real
>
class CudaVector
{
public:

    // -------------- //
    //  Type Aliases  //
    // -------------- //

    using RealType   = Real;
    using VectorType = CudaVector<RealType>;
    using MatrixType = CudaMatrix<RealType>;
    using ResultType = CudaVector<RealType>;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    /*! Performs a deep copy of the CudaVector object. */
    CudaVector(const CudaVector &);
    /*! Performs a deep copy of the CudaVector object. */
    CudaVector & operator=(const CudaVector &);

    /*! Performs a shallow copy of the CudaVector object. */
    CudaVector(CudaVector &&)                   = default;
    /*! Performs a shallow copy of the CudaVector object. */
    CudaVector & operator=(      CudaVector &&) = default;

    /** The constructor takes as an additional argument a reference to a
     *  CudaDevice object. This object handles device specific data and settings.
     *  Currently on the default_device is supported. */
    CudaVector(CudaDevice & = Cuda::default_device);
    /** Constructs a CudaVector object from a given VectorData object. The elements
     *  of the VectorData object are copied synchronously to the device */
    CudaVector(const VectorData<Real> & vector, CudaDevice & = Cuda::default_device);
    /** Constructs a CudaVector object from a given VectorData object. The elements
     *  of the VectorData object are copied synchronously to the device */
    CudaVector(VectorData<Real> && vector,      CudaDevice & = Cuda::default_device);

    // ----------------- //
    //   Manipulations   //
    // ----------------- //

    /** Resize the Cuda vector. This will allocate a new, uninitialized
     *  array on the device. */
    void resize(size_t i);

    /** Returns a device pointer to the element array. */
          RealType * get_element_pointer()       {return *elements;}
    /** Returns a device pointer to the element array. */
    const RealType * get_element_pointer() const {return *elements;}

    size_t rows() const {return n;}

    // ------------ //
    //  Arithmetic  //
    // ------------ //

    void accumulate(const CudaVector &v);
    void accumulate(RealType c);
    void subtract(const CudaVector &v);
    void scale(RealType c);

    RealType norm() const;

    friend RealType dot<>(const CudaVector&, const CudaVector&);

    // ------------- //
    //  Conversions  //
    // ------------- //

    operator VectorData<Real>() const;

private:

    size_t n;

    std::shared_ptr<RealType *> elements;

    /*! Cuda device object that holds device specific handles and data. */
    CudaDevice *    device;
    /*! Cuda allocator that manages the vector memory. */
    CudaAllocator * allocator;

    /*! Custom shared_ptr Deleter
     *
     * Deleter that correctly frees the CudaVector memory by calling
     * the release() method of the corresponding CudaAllocator object
     */
    struct CudaVectorDeleter
    {
        CudaVectorDeleter(CudaAllocator * allocator_) : allocator(allocator_) {}
        CudaVectorDeleter(const CudaVectorDeleter & )             = default;
        CudaVectorDeleter(      CudaVectorDeleter &&)             = default;
        CudaVectorDeleter & operator=(const CudaVectorDeleter  &) = default;
        CudaVectorDeleter & operator=(      CudaVectorDeleter &&) = default;

        void operator() (Real ** ptr)
        {
            if (* ptr) {
                allocator->release(* ptr);
            }
            delete ptr;
        }
        CudaAllocator * allocator;
    } deleter;
};

#include "cuda_vector.cpp"

}      // namespace invlib
#endif // CUDA_CUDA_VECTOR_H
