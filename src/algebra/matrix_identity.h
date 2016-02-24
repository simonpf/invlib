#ifndef ALGEBRA_MATRIX_IDENTITY_H
#define ALGEBRA_MATRIX_IDENTITY_H

#include <iostream>

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
typename Real,
typename Matrix
>
class MatrixIdentity
{

public:

    using MatrixBase = Matrix;

    // ----------------- //
    //   Constructors    //
    // ----------------- //

    MatrixIdentity() : c(1.0) {}

    MatrixIdentity( Real c_ ) : c(c_) {}

    // ------------------ //
    //      Addition      //
    // ------------------ //

    Matrix add(const Matrix& B) const
    {
        Matrix C(B);

        for (unsigned int i = 0; i < B.cols(); i++)
        {
            C(i,i) += c;
        }

        return C;
    }

    // ------------------ //
    //   Multiplication   //
    // ------------------ //

    template <typename T>
    T multiply(const T& B) const
    {
        return T(c * B); // Use perfect forwarding here!
    }

    const Real& scale() const
    {
        return c;
    }

    // ----------------- //
    // Addition Operator //
    // ----------------- //

    template <typename T>
    using Sum = MatrixSum<MatrixIdentity, T, Matrix>;

    template<typename T>
    Sum<T> operator+(const T &B)
    {
        return Sum<T>(*this, B);
    }

    // ----------------------- //
    // Multiplication Operator //
    // ----------------------- //

    template <typename T>
    using Product = MatrixProduct<MatrixIdentity, T, Matrix>;

    template<typename T>
    Product<T> operator*(const T &A) const
    {
        return Product<T>(*this, A);
    }

    MatrixIdentity invert() const
    {
        MatrixIdentity A(1.0 / c);
        return A;
    }

    template<typename Vector>
    Vector solve(const Vector& v) const
    {
        Vector w((1.0 / c) * v);
        return w;
    }

private:

    Real c;
};

template
<
typename Real,
typename Matrix
>
MatrixIdentity<Real, Matrix> operator*(Real c,
                                       const MatrixIdentity<Real, Matrix> &I)
{
    return MatrixIdentity<Real, Matrix>(c * I.scale());
}

template
<
typename T
>
auto operator*(double c,
               const T& B)
    -> typename MatrixIdentity<double, typename T::MatrixBase>::template Product<T>
{

    using Matrix = typename T::MatrixBase;
    using I = MatrixIdentity<double, Matrix>;
    using P = typename MatrixIdentity<double, Matrix>::template Product<T>;

    return P(I(c), B);
}
#endif //ALGEBRA_MATRIX_IDENTITY_H
