/** \file algebra/matrix_sum.h
 *
 * \brief Proxy class for computing the sum of two algebraic expressions.
 *
 */

#ifndef ALGEBRA_SUM_H
#define ALGEBRA_SUM_H

#include <type_traits>

namespace invlib
{

// -------------------- //
// Forward Declarations //
// -------------------- //

template
<
typename T1,
typename T2
>
class MatrixProduct;
template

<
typename T1,
typename T2,
typename Matrix
>
class MatrixDifference;

/** \brief Proxy class for computing the sum of two algebraic expressions.
 *
 * The MatrixSum class template provides a proxy class template for computing the
 * sum of two algebraic expressions. The class expects the left hand side
 * operand to be convertible to either a vector or a matrix, which then must
 * provide a member function accumulate, which can be called with the right
 * hand operand as only argument.
 *
 * \tparam T1 The type of the left hand side operand
 * \tparam T2 the type of the right hand side operand
 * \tparam Matrix The underlying matrix type used.
 *
 */
template
<
typename T1,
typename T2,
typename Matrix
>
class MatrixSum
{

public:

    using Real = typename Matrix::VectorBase::Real;
    using MatrixBase = Matrix;
    using VectorBase = typename Matrix::VectorBase;
    using Vector = VectorBase;

    MatrixSum(T1 Op1, T2 Op2)
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
    using Product = MatrixProduct<MatrixSum, T3>;

    template<typename T3>
    auto operator*(T3 &&C) const -> Product<T3>
    {
        return Product<T3>{*this, C};
    }

    // ------------------ //
    //  Nested  Addition  //
    // ------------------ //

    template <typename T3>
    using Sum = MatrixSum<MatrixSum, T3, Matrix>;

    template <typename T3>
    auto operator+(T3 &&C) const -> Sum<T3> const
    {
        return Sum<T3>(*this, C);
    }

    template <typename T3>
    using Difference = MatrixDifference<MatrixSum, T3, Matrix>;

    template <typename T3>
    auto operator-(T3 &&C) const -> Difference<T3> const
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
    T1 A;
    T2 B;

};

}

#endif // ALGEBRA_SUM_H
