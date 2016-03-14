/** \file algebra/matrix_transpose.h
 *
 * \brief Proxy class for computing the transpose of a matrix.
 *
 * Contains the invlib::MatrixTranspose class template with provides a proxy
 * class for transposing an algebraic expression.
 *
 * Also provides the function template transp(), to compute the transpose of
 * an algebraic expression.
 *
 */

#ifndef ALGEBRA_MATRIX_TRANSPOSE_H
#define ALGEBRA_MATRIX_TRANSPOSE_H

namespace invlib
{

/** \brief Proxy class for computing the transpose of a matrix.
 *
 * The MatrixTranspose class template provides a template proxy class
 * for computing the transpose of an algebraic expression.
 *
 * The class assumes that the algebraic expression provides the following
 * member functions:
 *
 * - Matrix transpose()
 * - Matrix transpose_multiply(const Matrix&)
 * - Matrix transpose_multiply(const Vector&)
 * - Matrix transpose_add(const Vector&)
 *
 * \tparam T1 The type of the algebraic expression to invert.
 * \tparam Matrix The underlying matrix type.
 *
 */
template
<
typename T1,
typename Matrix
>
class MatrixTranspose
{

public:

    using MatrixBase = Matrix;
    using VectorBase = typename Matrix::VectorBase;
    using Vector = VectorBase;

    // ----------------- //
    //   Constructors    //
    // ----------------- //

    MatrixTranspose(T1 A_)
        : A(A_) {}

    // ------------------ //
    //      Addition      //
    // ------------------ //

    Matrix add(const Matrix& B)
    {
        Matrix C(static_cast<Matrix>(A).transpose_add(B));
        return C;
    }

    // ------------------ //
    //   Multiplication   //
    // ------------------ //

    Matrix multiply(const Matrix& B) const
    {

        Matrix C = static_cast<Matrix>(A).transpose_multiply(B);
        return C;
    }

    Vector multiply(const Vector& v) const
    {

        Vector w = static_cast<Matrix>(A).transpose_multiply(v);
        return w;
    }

    // ----------------- //
    // Addition Operator //
    // ----------------- //

    template <typename T>
    using Sum = MatrixSum<MatrixTranspose, T, Matrix>;

    template<typename T>
    Sum<T> operator+(T &&B)
    {
        return Sum<T>(*this, B);
    }

    // ----------------------- //
    // Multiplication Operator //
    // ----------------------- //

    template <typename T>
    using Product = MatrixProduct<MatrixTranspose, T>;

    template<typename T>
    Product<T> operator*(T &&A) const
    {
        return Product<T>(*this, A);
    }

    Matrix invert() const
    {
        Matrix B(static_cast<Matrix>(A).transpose().invert());
        return B;
    }

    Vector solve(const Vector& v) const
    {
        Matrix B(static_cast<Matrix>(A).transpose().solve(v));
        return B;
    }

    operator Matrix() const
    {
        Matrix tmp = A.transpose();
        return tmp;
    }

private:

    T1 A;

};

/** \brief Transpose of an algebraic expression
 *
 * Returns a MatrixTranspose object instantiated with the type of the given
 * algebraic expression
 *
 * \tparam T The type of the algebraic expression to invert.
 * \return The MatrixTranspose proxy object representing the transpose
 * of the given algebraic expression.
 *
 */
template
<
typename T
>
MatrixTranspose<T, typename decay<T>::MatrixBase> transp(T &&A)
{
    return MatrixTranspose<T, typename decay<T>::MatrixBase>(A);
}

}

#endif // ALGEBRA_MATRIX_TRANSPOSE_H
