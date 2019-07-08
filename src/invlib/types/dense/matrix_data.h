/**
 * \file dense/matrix_data.h
 *
 * Generic class that manages data for dense matrics stored contiguously
 * in memory in column-major order.
 *
 */
#ifndef DENSE_MATRIX_DATA_H
#define DENSE_MATRIX_DATA_H

#include "invlib/utility/array.h"

namespace invlib
{

/**
 * \brief General matrix data class
 *
 * This class serves as a general representation of matrix data.
 *
 * The data is held in a contiguous array of heap memory allocated using
 * malloc. The memory is managed using a shared_ptr holding the reference
 * to the array.
 *
 * \tparam ScalarType The floating point type used for the representation of
 * matrix elements.
 */
template
<
typename ScalarType
>
class MatrixData
{
public:

    template<typename SType2>
    using MData = MatrixData<SType2>;

    template<typename SType2>
    using VData = MatrixData<SType2>;

    // ----------------- //
    //   Constructors    //
    // ----------------- //

    MatrixData() = default;
    MatrixData(size_t m_, size_t n_, std::shared_ptr<ScalarType[]> elements_)
        : m(m_), n(n_), elements(elements_)
    {
        // Nothing to do here.
    }

    MatrixData(const MatrixData &other) = default;
    MatrixData(MatrixData &&)           = default;
    MatrixData& operator=(const MatrixData &other) = default;
    MatrixData& operator=(MatrixData &&)           = default;
    ~MatrixData() = default;

    MatrixData get_block(size_t i, size_t j,
                         size_t di, size_t dj) const {
        MatrixData block{};
        block.resize(di, dj);

        for (size_t k = i; k < i + di; ++k) {
            std::copy(elements + n * k + j,
                      elements + n * k + j + dj,
                      block.elements + dj * (k - i));
        }
        return block;
    }

    ScalarType * get_element_pointer() {
        return elements.get();
    }

    const ScalarType * get_element_pointer() const {
        return elements.get();
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
        elements = array::create<ScalarType>(i * j);
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

    size_t rows() const {
        return m;
    }

    size_t cols() const {
        return n;
    }

protected:

    size_t m   = 0;
    size_t n   = 0;
    bool owner = false;
    std::shared_ptr<ScalarType[]> elements;

};

}       // namespace invlib
#endif  // DENSE_MATRIX_DATA_H
