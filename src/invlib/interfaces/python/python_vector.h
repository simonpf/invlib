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

template<typename SType> class PythonMatrixData;

/*! Storage class for python vector data.
 *
 * This is essentially a wrapper class around Python vector data
 * provided in the form of contiguous floating point numbers in
 * memory.
 *
 * \tparam The floating point type used to represent scalars.
 */
template
<
typename ScalarType
>
class PythonVectorData
{
public:

    template<typename SType2>
    using MData = PythonMatrixData<SType2>;

    PythonVectorData() = default;
    PythonVectorData(ScalarType *elements, size_t n, bool copy);

    PythonVectorData(const PythonVectorData &);
    PythonVectorData(PythonVectorData &&);

    PythonVectorData& operator=(const PythonVectorData &);
    PythonVectorData& operator=(PythonVectorData &&);

    ~PythonVectorData();

    PythonVectorData get_block(size_t i,
                               size_t di) const;

    // ----------------- //
    //   Manipulations   //
    // ----------------- //

    /*! Resize vector.
     *
     * Resize the vector to an \f$i\f$ dimensional vector.
     *
     * \param i Number of rows of the resized matrix.
     */
    void resize(unsigned int i) {
        if (owner) {
            delete[] elements;
        }
        elements = new ScalarType[i];
        n = i;
    }

    /*! Element access.
     *
     * \param i Index of the element to access.
     */
    ScalarType & operator()(unsigned int i)
    {
        return elements[i];
    }

    /*! Read-only element access.
     *
     * \param i Index of the element to access.
     */
    ScalarType operator()(unsigned int i) const
    {
        return elements[i];
    }

    /*! Number of rows of the vector
     *
     * \return The number of rows (dimension) of the vector.
     */
    unsigned int rows() const
    {
        return n;
    }

    ScalarType * data_pointer(int i = 0);
    const ScalarType * data_pointer(int i = 0) const;

protected:

    unsigned int n  = 0;
    bool owner      = false;
    ScalarType * elements = nullptr;

};

// -----------------  //
//    Python Vector   //
// -----------------  //

template <typename SType> class PythonMatrix;

template <typename SType>
class PythonVector : public BlasVector<SType, PythonVectorData>
{
public:

    /*! The floating point type used to represent scalars. */
    using ScalarType   = SType;
    using RealType     = SType;
    /*! The fundamental vector type used for the matrix algebra.*/
    using VectorType = PythonVector<ScalarType>;
    /*! The fundamental matrix type used for the matrix algebra.*/
    using MatrixType = PythonMatrix<ScalarType>;
    using ResultType = PythonVector<ScalarType>;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    PythonVector() = default;

    PythonVector(ScalarType *elements, size_t n, bool copy);
    PythonVector(PythonVectorData<SType> const other) {
        PythonVectorData<SType>::operator=(other);
    }

    PythonVector(const PythonVector &) = default;
    PythonVector(PythonVector &&)      = default;

    PythonVector& operator=(const PythonVector &)  = default;
    PythonVector& operator=(PythonVector &&)       = default;

    ~PythonVector() = default;

    using BlasVector<SType, PythonVectorData>::n;
};


/*! Stream vector to string */
template <typename ScalarType>
std::ostream & operator<<(std::ostream &, const PythonVector<ScalarType>&);

#include "python_vector.cpp"

}      // namespace invlib
#endif // INTERFACES_PYTHON_PYTHON_MATRIX
