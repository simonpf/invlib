/**
 * \file interfaces/python/python_vector.h
 *
 * \brief Interface for numpy.ndarrays that
 * can be interpreted as vectors.
 *
 */
#ifndef INTERFACES_PYTHON_PYTHON_VECTOR
#define INTERFACES_PYTHON_PYTHON_VECTOR

#include <cassert>
#include <cmath>
#include <iterator>
#include <memory>
#include <iostream>
#include "Python.h"

namespace invlib
{

// -------------------- //
// Forward Declarations //
// -------------------- //

template
<
typename RealType
>
class PythonMatrix;

template
<
typename RealType
>
class PythonVector;

template
<
typename RealType
>
RealType dot(const PythonVector<RealType>&, const PythonVector<RealType>&);

// --------------------------  //
//   Python Vector Interface   //
// --------------------------  //

/*! Python vector interface
 *
 * This is a wrapper class around numpy.ndarrays to be interpreted as column
 * vectors. Vectors are assumed to be column vectors, that is a vector of length
 * n is required to have shape (n, 1) in numpy.
 *
 * \tparam The floating point type used to represent scalars.
 */
template
<
typename Real
>
class PythonVector
{
public:

    /*! The floating point type used to represent scalars. */
    using RealType   = Real;
    /*! The fundamental vector type used for the matrix algebra.*/
    using VectorType = PythonVector<RealType>;
    /*! The fundamental matrix type used for the matrix algebra.*/
    using MatrixType = PythonMatrix<RealType>;
    using ResultType = PythonVector<RealType>;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    PythonVector() = default;

    PythonVector(RealType *data, size_t n, bool copy);

    PythonVector(const PythonVector &);

    PythonVector(PythonVector &&) = default;

    PythonVector& operator=(const PythonVector &);

    /*! Move assignment operator.
     *
     * Move a given vector into this vector by deep copy of its element.
     *
     */
    PythonVector& operator=(PythonVector &&) = default;
    ~PythonVector();

    PythonVector get_block(unsigned int i,
                           unsigned int di) const;

    // ----------------- //
    //   Manipulations   //
    // ----------------- //

    /*! Resize vector.
     *
     * Resize the vector to an \f$i\f$ dimensional vector.
     *
     * \param i Number of rows of the resized matrix.
     */
    void resize(unsigned int i);

    /*! Element access.
     *
     * \param i Index of the element to access.
     */
    RealType & operator()(unsigned int i);

    /*! Read-only element access.
     *
     * \param i Index of the element to access.
     */
    RealType operator()(unsigned int i) const;

    /*! Number of rows of the vector
     *
     * \return The number of rows (dimension) of the vector.
     */
    unsigned int rows() const;

    RealType * data_pointer(int i = 0);
    const RealType * data_pointer(int i = 0) const;

    // ------------ //
    //  Arithmetic  //
    // ------------ //

    /*! Accumulate into vector.
     *
     * Element-wise add another vector to this vector.
     *
     * \param v The vector to accumate into this one.
     */
    void accumulate(const PythonVector &v);

    /*! Accumulate into vector.
     *
     * Add scalar to all elements in vector. This function is required
     * if the diagonal of a sum involving an identity matrix is to be
     * computed, which is the case if a Jacobian preconditioner is used
     * for the Levenberg-Marquardt method with an identity damping matrix.
     *
     * \param v The scalar to add to each element.
     */
    void accumulate(RealType c);

    /*! Subtract from vector.
     *
     * Element-wise subtract another vector from this one.
     *
     * \param v The vector to subtract from this one.
     */
    void subtract(const PythonVector &v);

    /*! Scale vector.
     *
     * Multiply each element by a scalar factor.
     *
     * \param c The factor c to scale the vector with.
     */
    void scale(RealType c);

    /*! Elementwise product of this vector and another vector.
     *
     * \param v The vector to elementwise multiply this vector with.
     */
    PythonVector element_multiply(const PythonVector & v) const;

    /*! Elementwise inverse of the vector.
     *
     * Sets all elements in the vector to their reciprocal.
     */
    void element_invert();

    /*! Dot product of two vectors
     *
     * \return The dot product \f$ \sum_{i=1}^n v_i w_i \f$ of the
     * two given vectors.
     */
    friend RealType dot<RealType>(const PythonVector&, const PythonVector&);

    /*! Euclidean norm of a vector.
    *
    * \return The Euclidean norm of this vector.
    */
    RealType norm() const;

private:

    unsigned int n  = 0;
    bool owner      = false;
    RealType * data = nullptr;

};

/*! Stream vector to string */
template <typename RealType>
std::ostream & operator<<(std::ostream &, const PythonVector<RealType>&);

#include "python_vector.cpp"

}      // namespace invlib
#endif // ARCHETYPES_VECTOR_ARCHETYPE_H
