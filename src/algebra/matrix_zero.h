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
    using VectorBase = typename Matrix::VectorBase;
    using Vector = VectorBase;

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
        T t = B;
        t *= 0.0;
        return t;
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
        Vector tmp; tmp.resize(v.rows());
        tmp *= 0.0;
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

template
<
typename T
>
MatrixZero<T> inv( const MatrixZero<T> &A )
{
    return MatrixZero<T>{};
}

#endif // ALGEBRA_MATRIX_ZERO_H
