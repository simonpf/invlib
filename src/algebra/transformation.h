/** \file algebra/transformation.h
 *
 * \brief Contains the Transformation class template representing generic
 * coordinate transformation.
 *
 * Also provides two concrete transformation classes, one being the identity
 * transformation and the other the scaling by the reciprocal of the square
 * root of the diagonal elements of a matrix.
 *
 */

#ifndef ALGEBRA_TRANSFORMATION
#define ALGEBRA_TRANSFORMATION

#include <cmath>
#include <utility>
#include <iostream>

namespace invlib
{

template
<
typename T1,
typename T2
>
class MatrixProduct;

template
<
typename T1,
typename T2
>
class MatrixSum;

/**
 * \brief Proxy class for transfomations.
 *
 * This class is used to defer applications of transformations to matrix
 * algebraic expression. Holds a reference to the transformation object and
 * applies either the void apply_vector(Vector&) or the void apply_matrix(Matrix&)
 * member methods, depending if the algebraic expression is evaluated to a
 * vector or to a matrix. The vector type Vector must be provided by the matrix
 * type via Matrix::VectorBase.
 *
 * \tparam T1 The type of the algebraic expression.
 * \tparam Matrix The underlying matrix type.
 * \tparam Transform The tansformation type implementing the transformation.
 */
template
<
typename T1,
typename Matrix,
typename Transform
>
class Transformation
{

public:

    using MatrixBase = Matrix;
    using Vector = typename Matrix::VectorBase;

    Transformation(const T1 A_, const Transform&& t_)
        : A(A_), t(t_) {}

    operator Vector() const
    {
        Vector tmp = A;
        t.apply_vector(tmp);
        return tmp;
    }

    operator Matrix() const
    {
        Matrix tmp = A;
        t.apply_matrix(tmp);
        return tmp;
    }

    Matrix invert() const
    {
        Matrix tmp = *this;
        return tmp.invert();
    }

    Vector solve(const Vector & v) const
    {
        Matrix tmp = *this;
        return tmp.solve(v);
    }

    // -------------------- //
    // Arithmetic Operators //
    // -------------------- //

    template <typename T2>
        using Sum = MatrixSum<Transformation, T2>;

    template<typename T2>
    Sum<T2> operator+(T2 &&B) const
    {
        return Sum<T2>{*this, B};
    }

    template <typename T2>
    using Difference = MatrixDifference<Transformation, T2, Matrix>;

    template <typename T2>
    auto operator-(T2 &&C) const -> Difference<T2> const
    {
        return Difference<T2>(*this, C);
    }

    template <typename T2>
        using Product = MatrixProduct<Transformation, T2>;

    template<typename T2>
    Product<T2> operator*(T2 &&B) const
    {
        return Product<T2>{*this, B};
    }

private:

    T1 A;
    const Transform& t;
};

/**
 * \brief The identity transformation.
 */
class Identity
{
public:

    /**
    * \brief Apply identity.
    */
    template <typename T>
    constexpr auto apply(T&& t) const
        -> decltype(std::forward<T>(t))
    {
        return std::forward<T>(t);
    }
};

/**
 * \brief Transform to normalize diagonal of given matrix.
 *
 * When applied to a vector, this transformation scales each component
 * by the reciprocal square root of the absolute value of the corresponding
 * diagonal element  of the matrix @A_. When applied to a matrix, each row and
 * each column are scaled by the reciprocal of the square root of the diagonal
 * elements. When applied to the matrix A itself, the resulting matrix will
 * have only +/- 1.0 on the diagonal.
 *
 * \tparam Matrix The type of the matrix @A_. Must declare the associated vector
 * type as VectorBase.
 */
template
<
typename Matrix
>
class NormalizeDiagonal
{

public:

    using Vector = typename Matrix::VectorBase;

    NormalizeDiagonal(const Matrix &A_)
        : A(A_) {}

    void apply_matrix(Matrix &B) const
    {
        unsigned int m, n;
        m = B.rows();
        n = B.cols();

        for (unsigned int i = 0; i < m; i++)
        {
            for (unsigned int j = 0; j < n; j++)
            {
                B(i,j) *= 1.0 / sqrt(A(i,i) * A(j,j));
            }
        }
    }

    void apply_vector(Vector &v) const
    {
        unsigned int m;
        m = v.rows();

        for (unsigned int i = 0; i < m; i++)
        {
            v(i) *= 1.0 / sqrt(A(i,i));
        }
    }

    template <typename T>
    using Transform = Transformation<T, Matrix, NormalizeDiagonal&>;

    template<typename T>
    Transform<T> apply(const T& A)
    {
        return Transform<T>(A, *this);
    }

private:

    const Matrix& A;

};

}

#endif // ALGEBRA_TRANSFORMATION
