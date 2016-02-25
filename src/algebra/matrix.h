#ifndef ALGEBRA_MATRIX
#define ALGEBRA_MATRIX

#include "matrix_product.h"
#include "matrix_inverse.h"
#include "matrix_sum.h"
#include "matrix_difference.h"
#include "matrix_identity.h"

/**
 * \brief Wrapper class for symbolic matrix computations.
 *
 * The Matrix class template provides an abstraction layer for delayin matrix
 * computations on a given base matrix type Base and a corresponding vector type
 * Vector. The following delayed operations are provided:
 *
 * - Addition: operator+()
 * - Subtraction: operator-()
 * - Multiplication: operator*()
 * - Computing the inverse matrix: inv()
 * - Solving the system for a given vector: solve()
 * - Transpose: transp(const Matrix&)
 *
 * All delayed operations return proxy classes representing the computation,
 * which contain only reference to the matrices and vectors holding the actual
 * data. The execution of the computation is triggered when the proxy class
 * is converted to the given base matrix type or vector. During the computation
 * the operations are forwarded to the base matrix type.
 *
 * \tparam Base The base matrix type.
 * \tparam Vector The corresponding vector type.
 */
template
<
typename Base,
typename Vector
>
class Matrix : public Base
{

public:

    using VectorBase = Vector;
    using MatrixBase = Matrix;

    // -------------------- //
    //     Constructors     //
    // -------------------- //

    Matrix()
        : Base() {}

    // Copy constructors.
    Matrix(const Matrix& B) = default;
    Matrix(Matrix&& B)      = default;

    // Assignent operators.
    Matrix& operator=(const Matrix&) = default;
    Matrix& operator=(Matrix&& B)    = default;

    // Copy from base.
    Matrix(const Base& B)
        : Base(B) {}

    Matrix(const Base&& B)
        : Base(B) {}

    operator const Matrix()
    {
        return *this;
    }

    // ----------------- //
    //   Multiplication  //
    // ----------------- //

    Matrix multiply(const Matrix& B) const
    {
        Matrix C(this->Base::operator*(B));
        return C;
    }

    Vector multiply(const Vector& v) const
    {
        Vector w(this->Base::operator*(v));
        return w;
    }

    // ---------- //
    //   Scaling  //
    // ---------- //

    template <typename Real>
    Matrix scale(Real c) const
    {
        Matrix C(this->Base::operator*(c));
        return C;
    }

    // ----------------- //
    // Addition operator //
    // ----------------- //

    Matrix add(const Matrix &B) const
    {
        Matrix C = static_cast<Matrix>(this->Base::operator+(B));
        return C;
    }

    void subtract(const Matrix &B)
    {
        this->Base::operator-=(B);
    }

    template<typename Real>
    void accum(const MatrixIdentity<Real,Matrix> &B)
    {
        for (unsigned int i = 0; i < this->rows(); i++)
        {
            (*this)(i,i) += B.scale();
        }
    }

    void accum(const Matrix &B)
    {
        this->operator+=(B);
    }

    template <typename T>
        using Sum = MatrixSum<Matrix, T, Matrix>;

    template<typename T>
    Sum<T> operator+(const T &B) const
    {
        return Sum<T>{*this, B};
    }

    template <typename T>
    using Difference = MatrixDifference<Matrix, T, Matrix>;

    template <typename T>
    auto operator-(const T &C) const -> Difference<T> const
    {
        return Difference<T>(*this, C);
    }

    // ----------------------- //
    // Multiplication operator //
    // ----------------------- //

    template <typename T>
        using Product = MatrixProduct<Matrix, T, Matrix>;

    template<typename T>
    Product<T> operator*(const T &B) const
    {
        return Product<T>{*this, B};
    }

};

#endif // ALGEBRA_MATRIX
