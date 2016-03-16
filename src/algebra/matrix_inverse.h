#ifndef ALGEBRA_MATRIX_INVERSE
#define ALGEBRA_MATRIX_INVERSE

/** \file algebra/matrix_inverse.h
 *
 * \brief Proxy class for computing matrix inverses.
 *
 * Contains the class invlib::MatrixInverse which is a proxy class
 * for computing the inverse of a matrix or solving the corresponding
 * system of equations.
 *
 * Also contains the generic inv() function which creates a MatrixInverse
 * object from a given algebraic expression.
 *
 */

#include <traits.h>

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
typename T2
>
class MatrixSum;

/** \brief Proxy class for computing matrix inverses.
 *
 * The MatrixDifference class template provides a template proxy class
 * for computing the inverses of matrices or solving the corresponding
 * linear system of equations.
 *
 * The key purpose of the class is to delay the computation of the inverse
 * until it is either multiplied by a vector or used in another algebraic
 * operation (including multiplication by another matrix). If the MatrixInverse
 * object is multiplied by a vector only the corresponding linear system
 * has to be solved. All other operations require the inversion of the matrix.
 *
 * A MatrixInverse object can be used in another operation or converted to
 * an object of type Matrix. To this end, the T1 type is required to provide
 * the member functions
 *
 * - Vector solve(const typename Vector&)
 * - Matrix invert()
 *
 * for solving the associated linear system of equations and inverting the
 * corresponding matrix, respectively.
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
class MatrixInverse
{
public:

    using MatrixBase = Matrix;
    using VectorBase = typename Matrix::VectorBase;
    using Vector = VectorBase;

    MatrixInverse(T1 A_)
        : A(A_) {}

    operator Matrix() const
    {
        Matrix B = A.invert();
        return B;
    }

    // ----------------- //
    //     Addition      //
    // ----------------- //

    Matrix add(const Matrix &B) const
    {
        Matrix C = A.invert() + B;
        return C;
    }

    // ----------------- //
    //   Multiplication  //
    // ----------------- //

    Vector multiply(const Vector &v) const
    {
        Vector w = A.solve(v);
        return w;
    }

    Matrix multiply(const Matrix &B) const
    {
        Matrix C = A.invert() * B;
        return C;
    }

    // -------------------------- //
    //   Multiplication Operator  //
    // -------------------------- //

    template<typename T>
        using Product = MatrixProduct<MatrixInverse, T>;

    template <typename T>
    auto operator*(T &&B) const -> Product<T>
    {
        return Product<T>(*this, B);
    }

    // --------------------- //
    //   Addition  Operator  //
    // --------------------- //

    template<typename T>
    using Sum = MatrixSum<MatrixInverse, T>;

    template <typename T>
    auto operator+(T &&B) const -> Sum<T>
    {
        return Sum<T>(*this, B);
    }

private:

    T1 A;

};

/** \brief Inverse of an algebraic expression.
 *
 * Creates a proxy object of type
 * MatrixInverse<T, typename T::MatrixBase> will evaluate to either
 * inversion of the corresponding matrix or solution of the corresponding
 * linear system.
 *
 * If the resulting MatrixInverse object is multiplied from the right with
 * a vector the corresponding linear system will only be solved. All other
 * operations will result in the inversion of the corresponding system.
 *
 * \tparam T The type of the algebraic expression.
 *
 * \param A The algebraic expression to be inverted.
 *
 * \return The MatrixInverse object representing the inverted algebraic
 * expression.
 *
 */
template
<
typename T
>
    MatrixInverse<T, typename decay<T>::MatrixBase> inv(T &&A)
{
    return MatrixInverse<T, typename decay<T>::MatrixBase>(A);
}

}

#endif // ALGEBRA_MATRIX_INVERSE

