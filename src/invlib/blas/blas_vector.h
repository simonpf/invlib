/**
 * \file blas/blas_vector.h
 *
 * \brief Dense vector arithmetic using BLAS.
 *
 */
#ifndef BLAS_BLAS_VECTOR_H
#define BLAS_BLAS_VECTOR_H

#include "invlib/dense/vector_data.h"
#include "invlib/sparse/sparse_data.h"
#include "invlib/blas/blas_generic.h"

namespace invlib {

// -------------------- //
// Forward Declarations //
// -------------------- //

template <typename RealType> class BlasVector;

template <typename Real> class BlasMatrix
{
public:
    using RealType   = Real;
    using VectorType = BlasVector<Real>;
    using MatrixType = BlasMatrix;
    using ResultType = BlasVector<Real>;
};

template <typename RealType>
RealType dot(const BlasVector<RealType>&, const BlasVector<RealType>&);

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
 * \tparam Real Floating point type used for the representation of
 * the vector elements.
 */
template
<
typename Real
>
class BlasVector : public VectorData<Real>
{
public:

    // -------------- //
    //  Type Aliases  //
    // -------------- //

    using RealType   = Real;
    using VectorType = BlasVector;
    using MatrixType = BlasMatrix<RealType>;
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

    /** Constructs a BlasVector object from a given VectorData object. Only
     * performs a shallow copy meaning that the VectorData object and the
     * BlasVector share the same vector data. */
    BlasVector(const VectorData<Real> & v);
    /** Constructs a BlasVector object from a given VectorData object. */
    BlasVector(VectorData<Real> && v);

    // ------------ //
    //  Arithmetic  //
    // ------------ //

    void accumulate(const BlasVector &v);
    void accumulate(RealType c);
    void subtract(const BlasVector &v);
    void scale(RealType c);

    RealType norm() const;

    friend RealType dot<>(const BlasVector&, const BlasVector&);

private:

    // ------------------- //
    //  Base Class Members //
    // ------------------- //

    using VectorData<Real>::elements;
    using VectorData<Real>::n;

};

#include "blas_vector.cpp"

}      // namespace invlib
#endif // BLAS_BLAS_VECTOR_H
