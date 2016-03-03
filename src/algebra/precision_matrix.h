#ifndef PRECISION_MATRIX_H
#define PRECISION_MATRIX_H

#include "algebra.h"
#include <iostream>

template
<
typename Matrix
>
class PrecisionMatrix;

template
<
typename T1,
typename T2,
typename Matrix
>
class MatrixProduct;

template
<
typename T1,
typename T2,
typename Matrix
>
class MatrixSum;

template
<
typename Matrix
>
const Matrix& inv(const PrecisionMatrix<Matrix> &A);

/**
 * \brief Wrapper to represent precision matrices.
 *
 * The computation of MAP estimators involves covariance matrices mainly
 * through their inverse. In some cases it may thus be beneficial to provide
 * the inverse directly. Using the PrecisionMatrix type to instantiate the
 * MAP class will interpret the matrices given to the MAP constructor as
 * precision matrices.
 *
 * Technically this is achieved by overloading the inv() function as well
 * as overriding all relevant member functions used for the matrix algebra.
 * The class provides the overloaded inv() function to return a reference to
 * the actual precision matrix provided. Operations involving the wrapper
 * directly, however, require inverting the matrix (or at least solving the
 * the correspongind linear system).
 *
 * Also provides an overloaded transp() function, that does nothing, since the
 * a precision matrix is by definition symmetric.
 *
 */
template
<
typename Matrix
>
class PrecisionMatrix
{

public:

    using Vector = typename Matrix::VectorBase;

    template <typename T>
    using Product = MatrixProduct<PrecisionMatrix, T, Matrix>;

    template <typename T>
    Product<T> operator*(const T& B) const
    {
        return Product<T>(*this, B);
    }

    template <typename T>
    using Sum = MatrixSum<PrecisionMatrix, T, Matrix>;

    template <typename T>
    Product<T> operator+(const T& B) const
    {
        return Sum<T>(*this, B);
    }

    PrecisionMatrix(const Matrix& A_)
        : A(A_) {}


    Matrix multiply(const Matrix& B) const
    {
        Matrix tmp = A.invert();
        return tmp * B;
    }

    Vector multiply(const Vector& v) const
    {
        return A.solve(v);
    }

    operator Matrix() const
    {
        Matrix tmp = A.invert();
        return tmp;
    }

    friend const Matrix& inv<Matrix>(const PrecisionMatrix<Matrix> &A);

private:

    const Matrix& A;
};

/** \brief Inversion of precision matrices.
 *
 * Simply returns the reference to the precision matrix that is contained in
 * the PrecisionMatrix wrapper.
 *
 * \tparam Matrix The type of the underlying precision matrix.
 * \param A The precision matrix to invert.
 *
 */
template
<
typename Matrix
>
const Matrix& inv(const PrecisionMatrix<Matrix> &A)
{
    return A.A;
}

#endif // PRECISION_MATRIX_H
