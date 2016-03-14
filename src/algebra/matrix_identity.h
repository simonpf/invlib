#ifndef ALGEBRA_MATRIX_IDENTITY_H
#define ALGEBRA_MATRIX_IDENTITY_H

#include <iostream>
#include <traits.h>

/** \file algebra/matrix_identity.h
 *
 * \brief Generic identity matrix.
 *
 * Contains the class invlib::Matrixidentity which implements a generic
 * identity matrix and overloads for relevant algebraic operations.
 *
 */

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
class MatrixSum;

/** \brief Generic matrix identity.
 *
 * A class template representing a (scaled) matrix identity matrix.
 *
 * \tparam Real The floating point type used for scalars.
 * \tparam Matrix The underlying Matrix type that is used.
 *
 */
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

    MatrixIdentity(Real c_) : c(c_) {}

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

    MatrixIdentity add(const MatrixIdentity& B) const
    {
        return MatrixIdentity(c + B.c);
    }

    // ------------------ //
    //   Multiplication   //
    // ------------------ //

    template <typename T>
    T multiply(const T& B) const
    {
        return B.scale(c); // Use perfect forwarding here!
    }

    const Real& scale() const
    {
        return c;
    }

    // ---------- //
    //   Scaling  //
    // ---------- //

    MatrixIdentity scale(Real d) const
    {
        return MatrixIdentity(c * d); // Use perfect forwarding here!
    }

    // ----------------- //
    // Addition Operator //
    // ----------------- //

    template <typename T>
    using Sum = MatrixSum<MatrixIdentity, T, Matrix>;

    template<typename T>
    Sum<T> operator+(T &&B) const
    {
        return Sum<T>(*this, B);
    }

    // ----------------------- //
    // Multiplication Operator //
    // ----------------------- //

    template <typename T>
    using Product = MatrixProduct<MatrixIdentity, T>;

    template<typename T>
    Product<T> operator*(T &&A) const
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

/** \brief Multiplication by a scalar.
 *
 * Overload of the * operator for multiplication of an algebraic
 * expression by a scalar.
 *
 * \tparam T The type of the algebraic expression.
 *
 * \param c The scaling factor.
 * \param B The algebraic expression to be scaled.
 *
 * \return A matrix product proxy object with a scaled identity matrix and
 * the given algebraic expression as operands.
 */
template
<
typename T,
typename Real = typename decay<T>::Real,
typename Matrix = typename decay<T>::MatrixBase,
typename = disable_if< is_same<decay<T>, MatrixIdentity<Real, Matrix> > >
>
auto operator*(double c, T&& B)
    -> MatrixProduct<MatrixIdentity<Real, Matrix>, T>

{
    using I = MatrixIdentity<Real, Matrix>;
    using P = typename I::template Product<T>;
    return P(I(c), B);
}

/** \brief Scale identity matrix.
 *
 * Overload of the * operator for multiplication of an identity
 * matrix by a scalar.
 *
 * \tparam T The underlying matrix type of the identity matrix object.
 *
 * \param c The scaling factor.
 * \param B The identity matrix object to be scaled.
 *
 * \return A scaled identity matrix object.
 *
 */
template
<
typename T,
typename Real = double
>
auto operator*(double c,
               const MatrixIdentity<Real, T>& B)
    -> MatrixIdentity<Real, T>
{
    return B.scale(c);
}

/** \brief Identity matrix inverse.
 *
 * Compute the inverse of a (scaled) identity matrix.
 *
 * \tparam T The underlying matrix type of the identity matrix object.
 *
 * \param B The identity matrix to be inverted.
 *
 * \return The inverted identity matrix object.
 *
 */
template
<
typename Real,
typename T
>
auto inv(const MatrixIdentity<Real,T> &A)
    -> MatrixIdentity<Real, typename T::MatrixBase>
{
    return MatrixIdentity<Real, T>(1.0 / A.scale());
}

}

#endif //ALGEBRA_MATRIX_IDENTITY_H

