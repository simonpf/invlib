#ifndef ALGEBRA_MATRIX_PRODUCT_H
#define ALGEBRA_MATRIX_PRODUCT_H

#include <iostream>
#include <type_traits>

template
<
typename T1,
typename T2,
typename Matrix
>
class MatrixSum;

template <typename T> void foo(T t);

template
<
typename T1,
typename T2,
typename Matrix
>
class MatrixProduct
{

public:

    using MatrixBase = Matrix;
    using VectorBase = typename Matrix::VectorBase;
    using Vector = VectorBase;

    template <typename T3>
    using NestedProduct = typename T2::template Product<T3>;

    template <typename T3>
    using Product = MatrixProduct<T1, NestedProduct<T3>, Matrix>;

    // ------------------ //
    //   Multiplication   //
    // ------------------ //

    Vector multiply(const Vector &v) const
    {
        Vector tmp1 = B.multiply(v);
        Vector tmp2 = A.multiply(tmp1);
        return tmp2;
    }

    Matrix multiply(const Matrix &C) const
    {
        Matrix tmp1 = B.multiply(C);
        Matrix tmp2 = A.multiply(tmp1);
        return tmp2;
    }

    // ------------------ //
    //      Addition      //
    // ------------------ //


    Matrix add(const Matrix &C) const
    {
        auto tmp1 = A.multiply(B);
        Matrix tmp2 = tmp1.add(C);
        return tmp2;
    }

    Vector add(const Vector &v) const
    {
        Vector tmp1 = B;
        Vector tmp2 = A * tmp1;
        tmp2 += v;
        return tmp2;
    }

    // ------------------ //
    //     Inversion      //
    // ------------------ //

    Matrix invert() const
    {
        Matrix tmp(static_cast<Matrix>(A).multiply(static_cast<Matrix>(B)));
        return tmp.invert();
    }

    Matrix solve(const Vector &v) const
    {
        Matrix tmp1(A);
        tmp1 += static_cast<Matrix>(B);
        return tmp1.solve(v);
    }


    MatrixProduct( const T1 &Op1, const T2 &Op2 )
        : A(Op1), B(Op2) {}

    template <typename T3>
    auto operator*(const T3 &C) const -> Product<T3>
    {
        return Product<T3>(A, B * C);
    }

    template <typename T3>
    using Sum = MatrixSum<MatrixProduct , T3, Matrix>;

    template <typename T3>
    auto operator+(const T3& C) const -> Sum<T3>
    {
        return Sum<T3>(*this, C);
    }

    /* template <typename T3> */
    /* auto operator+(const T3 &C) const -> Sum<T3> */
    /* { */
    /*     return typename Sum<T3>::type(A, B * C); */
    /* } */

    operator Matrix() const
    {
        Matrix tmp1 = B;
        Matrix tmp2 = A.multiply(tmp1);
        return tmp2;
    }

    operator Vector() const
    {
        Vector tmp1 = B;
        Vector tmp2 = A.multiply(tmp1);
        return tmp2;
    }

private:

    // Operand references.
    typedef typename
        std::conditional<std::is_same<T1, Vector>::value, const Vector&, T1>::type A1;

    typedef typename
        std::conditional<std::is_same<T1, Matrix>::value, const Matrix&, T1>::type A2;

    typedef typename
        std::conditional<std::is_same<T2, Vector>::value, const Vector&, T2>::type B1;

    typedef typename
        std::conditional<std::is_same<T2, Matrix>::value, const Matrix&, T2>::type B2;

    A2 A;
    B2 B;

};

#endif // ALGEBRA_MATRIX_PRODUCT_H
