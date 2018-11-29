/**
 * \file interfaces/python/python_matrix.h
 *
 * \brief Interface for numpy.ndarrays that
 * can be interpreted as dense matrices.
 *
 */
#ifndef INTERFACES_PYTHON_PYTHON_MATRIX
#define INTERFACES_PYTHON_PYTHON_MATRIX

#include <cassert>
#include <cmath>
#include <iterator>
#include <memory>
#include <iostream>

#include "invlib/interfaces/python/python_vector.h"
#include "invlib/blas/blas_matrix.h"

namespace invlib
{

// --------------------------  //
//     Python Matrix Data      //
// --------------------------  //

/*! Storage class for python matrix data.
 *
 * This class manages memory representing a two dimensional
 * dense matrix represented as contiguous data in memory.
 *
 * \tparam The floating point type used to represent scalars.
 */
template
<
typename ScalarType
>
class PythonMatrixData
{
public:

    template<typename SType2>
    using MData = PythonMatrixData<SType2>;

    template<typename SType2>
    using VData = PythonVectorData<SType2>;

    // ----------------- //
    //   Constructors    //
    // ----------------- //

    PythonMatrixData() = default;
    PythonMatrixData(ScalarType *elements_, size_t m_, size_t n_, bool copy)
        : m(m_), n(n_), owner(copy)
    {
        if (copy) {
            elements = new ScalarType[m * n];
            std::copy(elements_, elements_ + m_ * n_, elements);
        } else {
            elements = elements_;
        }
    }

    PythonMatrixData(const PythonMatrixData &other)
        : elements(other.elements), m(other.m), n(other.n), owner(false)
    {
        // Nothing to do here.
    }

    PythonMatrixData(PythonMatrixData &&) = default;

    PythonMatrixData& operator=(const PythonMatrixData &other) {
        elements = other.elements;
        m        = other.m;
        n        = other.n;
        owner    = other.owner;
    }
    PythonMatrixData& operator=(PythonMatrixData &&) = default;

    ~PythonMatrixData() {
        if (owner) {
            delete[] elements;
        }
    }

    PythonMatrixData get_block(size_t i, size_t j,
                               size_t di, size_t dj) const {
        PythonMatrixData block{};
        block.resize(di, dj);

        for (size_t k = i; k < i + di; ++k) {
            std::copy(elements + n * k + j,
                      elements + n * k + j + dj,
                      block.elements + dj * (k - i));
        }
        return block;
    }

    ScalarType * get_element_pointer() {
        return elements;
    }

    const ScalarType * get_element_pointer() const {
        return elements;
    }

    // ----------------- //
    //   Manipulations   //
    // ----------------- //

    /*! Resize the matrix.
     *
     * Resize the vector to an \f$i\f$ dimensional vector.
     *
     * \param i Number of rows of the resized matrix.
     * \param j Number of columns of the resized matrix.
     */
    void resize(size_t i, size_t j) {
        if (owner) {
            delete[] elements;
        }
        elements = new ScalarType[i * j];
        m = i; n = j;
    }

    //
    // Element access
    //

    ScalarType & operator()(size_t i, size_t j)
    {
        return elements[i * n + j];
    }
    ScalarType operator()(size_t i, size_t j) const
    {
        return elements[i * n + j];
    }

    //
    // Size of the matrix.
    //

    unsigned int rows() const
    {
        return m;
    }

    unsigned int cols() const
    {
        return n;
    }

protected:

    size_t m   = 0;
    size_t n   = 0;
    bool owner = false;
    ScalarType * elements = nullptr;

};

// -------------------  //
//    Python Matrix     //
// -------------------  //

template <typename SType>
class PythonMatrix : public BlasMatrix<SType, PythonMatrixData>
{
public:

    /*! The floating point type used to represent scalars. */
    using ScalarType   = SType;
    using RealType     = SType;
    using MatrixType   = PythonMatrix<ScalarType>;
    using ResultType   = PythonMatrix<ScalarType>;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    PythonMatrix() = default;

    PythonMatrix(const BlasMatrix<SType, PythonMatrixData> &data)
        : BlasMatrix<SType, PythonMatrixData>(data) {
        // Nothing to do here.
    }

    PythonMatrix(ScalarType *elements, size_t m, size_t n, bool copy) {
        PythonMatrixData<ScalarType>::operator=(
            PythonMatrixData<ScalarType>(elements, m, n, copy)
            );
    }

    PythonMatrix(const PythonMatrix &) = default;
    PythonMatrix(PythonMatrix &&)      = default;

    PythonMatrix& operator=(const PythonMatrix &)  = default;
    PythonMatrix& operator=(PythonMatrix &&)       = default;

    ~PythonMatrix() = default;

    using BlasMatrix<SType, PythonMatrixData>::n;
};

}      // namespace invlib
#endif // INTERFACES_PYTHON_PYTHON_MATRIX
