/**
 * \file blas/blas_vector.h
 *
 * \brief Dense vector arithmetic using BLAS.
 *
 */
#ifndef BLAS_BLAS_VECTOR_H
#define BLAS_BLAS_VECTOR_H

#include "invlib/types/dense/vector_data.h"
#include "invlib/types/blas/blas_generic.h"

namespace invlib {

// -------------------- //
// Forward Declarations //
// -------------------- //

template <typename SType, template <typename> typename VData> class BlasVector;
template <typename SType, template <typename> typename MData> class BlasMatrix;

template <typename SType, template <typename> typename VData>
    SType dot(const BlasVector<SType, VData>&,
              const BlasVector<SType, VData>&);

// -------------------  //
//   Blas Vector Class  //
// -------------------  //

/**
 * \brief Dense Vector Arithmetic using BLAS
 *
 * Implements a dense vector class that uses BLAS level 1 functions for vector
 * arithmetic. In addition to that the vector type provides functions for scaling
 * and addition of a constant, which are necessary for use with the CG solver.
 *
 * \tparam SType Floating point type used for the representation of
 * the vector elements.
 */
template
<
    typename SType,
    template <typename> typename VData = VectorData
>
class BlasVector : public VData<SType>
{
public:

    // -------------- //
    //  Type Aliases  //
    // -------------- //

    using RealType   = SType;
    using VectorType = BlasVector;
    using MatrixType = BlasVector;
    using ResultType = BlasVector;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    BlasVector()                                = default;
    /*! Performs a shallow copy of the BlasVector object. */
    BlasVector(const BlasVector &)              = default;
    BlasVector(BlasVector &&)                   = default;
    /*! Performs a shallow copy of the BlasVector object. */
    BlasVector & operator=(const BlasVector & ) = default;
    BlasVector & operator=(      BlasVector &&) = default;

    /*! Construct BlasVector object from given VData object.
     *
     * Simply forwards the copy constructor call to that of the super class.
     * Its behavior thus depends on the VData class.
     */
    BlasVector(const VData<SType> & v);

    /*! Construct BlasVector object from given VData object.
    *
    * Forwards the move constructor call to that of the super class.
    * Its behavior thus depends on the VData class.
    */
    BlasVector(VData<SType> &&v);

    // ------------ //
    //  Data access //
    // ------------ //

    SType * get_element_pointer() {
        return elements.get();
    }

    const SType * get_element_pointer() const {
        return elements.get();
    }

    // ------------ //
    //  Arithmetic  //
    // ------------ //

    void accumulate(const BlasVector &v);
    void accumulate(SType c);
    void subtract(const BlasVector &v);
    void scale(SType c);

    SType norm() const;

    friend SType dot<>(const BlasVector&, const BlasVector&);

protected:

    // ------------------- //
    //  Base Class Members //
    // ------------------- //

    using VData<SType>::elements;
    using VData<SType>::n;

};

#include "blas_vector.cpp"

}      // namespace invlib
#endif // BLAS_BLAS_VECTOR_H
