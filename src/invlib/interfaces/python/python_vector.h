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

#include "invlib/blas/blas_vector.h"

namespace invlib
{

// --------------------------  //
//     Python Vector Data      //
// --------------------------  //


// -----------------  //
//    Python Vector   //
// -----------------  //

template <typename ScalarType, typename IndexType> class PythonMatrix;

template <typename ScalarType>
class PythonVector : public BlasVector<ScalarType, VectorData>
{
public:

    /*! The floating point type used to represent scalars. */
    using RealType     = ScalarType;
    /*! The fundamental vector type used for the matrix algebra.*/
    using VectorType = PythonVector<ScalarType>;
    using ResultType = PythonVector<ScalarType>;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    PythonVector() = default;
    PythonVector(ScalarType * other_elements, size_t rows, bool copy);

    PythonVector(const PythonVector &) = default;
    PythonVector(PythonVector &&)      = default;
    PythonVector& operator=(const PythonVector &)  = default;
    PythonVector& operator=(PythonVector &&)       = default;
    ~PythonVector() = default;

    using BlasVector<ScalarType, VectorData>::n;
};


/*! Stream vector to string */
template <typename ScalarType>
std::ostream & operator<<(std::ostream &, const PythonVector<ScalarType>&);

#include "python_vector.cpp"

}      // namespace invlib
#endif // INTERFACES_PYTHON_PYTHON_MATRIX
