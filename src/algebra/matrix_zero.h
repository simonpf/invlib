#ifndef ALGEBRA_MATRIX_ZERO_H
#define ALGEBRA_MATRIX_ZERO_H

#include "matrix_inverse.h"

template
<
typename Matrix
>
class MatrixZero
{

public:

    using MatrixBase = Matrix;
    using Vector     = typename Matrix::Vector;

    // ----------------- //
    //   Constructors    //
    // ----------------- //

    MatrixZero() = default;
    MatrixZero(const MatrixZero&) = default;
    MatrixZero& operator= (const MatrixZero&) = default;

    // ------------------ //
    //      Addition      //
    // ------------------ //

    Matrix add(const Matrix& B) const
    {
        Matrix C(B);
        return C;
    }

    // ------------------ //
    //   Multiplication   //
    // ------------------ //

    template <typename T>
    T multiply(const T& B) const
    {
        return T(0.0 * B);
    }

    // ------------------ //
    //     Inversion      //
    // ------------------ //

    MatrixZero invert() const
    {
        return MatrixZero();
    }

    Vector solve(const Vector &v) const
    {
        Vector tmp(0.0 * v);
        return tmp;
    }

    // ----------------- //
    // Addition Operator //
    // ----------------- //

    template <typename T>
        using Sum = T;

    template<typename T>
    const T& operator+(const T &B)
    {
        return B;
    }

    // ----------------------- //
    // Multiplication Operator //
    // ----------------------- //

    template <typename T>
    using Product = MatrixProduct<MatrixZero, T, Matrix>;

    template <typename T>
    Product<T> operator*(const T &A) const
    {
        return Product<T>(*this, A);
    }

};

#endif // ALGEBRA_MATRIX_ZERO_H
