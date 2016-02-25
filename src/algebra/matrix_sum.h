#ifndef ALGEBRA_SUM_H
#define ALGEBRA_SUM_H

#include <type_traits>

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
class MatrixDifference;

template
<
typename T1,
typename T2,
typename Matrix
>
class MatrixSum
{

public:

    using MatrixBase = Matrix;
    using VectorBase = typename Matrix::VectorBase;
    using Vector = VectorBase;

    MatrixSum( const T1 &Op1, const T2 &Op2 )
        : A(Op1), B(Op2) {}

    // ------------------ //
    //   Multiplication   //
    // ------------------ //

    Vector multiply(const Vector &v) const
    {
        Vector tmp = B.multiply(v);
        tmp += A.multiply(v);
        return tmp;
    }

    Matrix multiply(const Matrix &C) const
    {
        Matrix tmp1 = A;
        tmp1.accum(B);
        Matrix tmp2 = tmp1.multiply(C);
        return tmp2;
    }

    // ------------------ //
    //      Addition      //
    // ------------------ //

    Matrix add(const Matrix &C) const
    {
        Matrix tmp = A;
        tmp.accum(B);
        tmp.accum(C);
        return C;
    }

    Matrix subtract(const Matrix &C) const
    {
        Matrix tmp = A;
        tmp.accum(B);
        tmp.accum(C);
        return tmp;
    }

    // ------------------ //
    //     Inversion      //
    // ------------------ //

    Matrix invert() const
    {
        Matrix tmp = A;
        tmp.accum(B);
        return tmp.invert();
    }

    Vector solve(const Vector &v) const
    {
        Matrix tmp = A;
        tmp.accum(B);
        return tmp.solve(v);
    }

    // ----------------------- //
    // Multiplication Operator //
    // ----------------------- //

    template <typename T3>
    using Product = MatrixProduct<MatrixSum, T3, Matrix>;

    template<typename T3>
    auto operator*(const T3& C) const -> Product<T3>
    {
        return Product<T3>(*this, C);
    }

    // ------------------ //
    //  Nested  Addition  //
    // ------------------ //

    template <typename T3>
    using Sum = MatrixSum<MatrixSum, T3, Matrix>;

    template <typename T3>
    auto operator+(const T3 &C) const -> Sum<T3> const
    {
        return Sum<T3>(*this, C);
    }

    template <typename T3>
    using Difference = MatrixDifference<MatrixSum, T3, Matrix>;

    template <typename T3>
    auto operator-(const T3 &C) const -> Difference<T3> const
    {
        return Difference<T3>(*this, C);
    }

    operator Matrix() const
    {
        Matrix tmp = A;
        tmp.accum(B);
        return tmp;
    }

    operator Vector() const
    {
        Vector tmp = A;
        tmp.accum(B);
        return tmp;
    }

private:

    // Operand references.
    typedef typename
        std::conditional<std::is_same<T1, VectorBase>::value, const VectorBase&, T1>::type A1;

    typedef typename
        std::conditional<std::is_same<T1, Matrix>::value, const Matrix&, T1>::type A2;

    typedef typename
        std::conditional<std::is_same<T2, VectorBase>::value, const VectorBase&, T2>::type B1;

    typedef typename
        std::conditional<std::is_same<T2, Matrix>::value, const Matrix&, T2>::type B2;

    A2 A;
    B2 B;

};

#endif // ALGEBRA_SUM_H
