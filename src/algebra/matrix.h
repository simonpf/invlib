#ifndef ALGEBRA_MATRIX
#define ALGEBRA_MATRIX

#include <type_traits>

#include "matrix_product.h"
#include "matrix_inverse.h"
#include "matrix_sum.h"
#include "matrix_difference.h"
#include "matrix_identity.h"
#include "matrix_zero.h"

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

    using Real       = typename Base::Real;
    using VectorBase = Vector;
    using MatrixBase = Matrix;
    using I = MatrixIdentity<Real, Matrix>;

    class ElementIterator;

    ElementIterator begin()
    {
        return ElementIterator(this);
    }

    ElementIterator end()
    {
        return ElementIterator();
    }

    // -------------------- //
    //     Constructors     //
    // -------------------- //

    template
    <
    typename T,
    typename = std::enable_if<!std::is_same<Matrix, std::decay<T>>::value>
    >
    Matrix( T && t)
        : Base(std::forward<T>(t)) {}

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

    void accum(const MatrixZero<Matrix> &Z) {}

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

/**
 * \brief Iterator for matrix element access.
 *
 * Provides an interface for element-wise access of matrix elements. Accesses
 * elements using <tt>Base::operator()(unsigned int, unsigned int)</tt>. Assumes
 * that the number of rows and columns in the matrix can be obtained from the
 * member functions <tt>rows()</tt> and <tt>cols()</tt>, respectively. Assumes
 * indexing to starting at 0.
 *
 * \tparam Base The base matrix type.
 * \tparam Vector The corresponding vector type.
 */
template
<
typename Base,
typename Vector
>
class Matrix<Base,Vector>::ElementIterator
{
public:

    using MatrixType = Matrix<Base,Vector>;

    ElementIterator() {}

    ElementIterator(MatrixType* M_)
        : M(M_), i(0), j(0), k(0), m(M_->rows()), n(M_->cols()) {}

    Real& operator*()
    {
        return M->operator()(i,j);
    }

    Real& operator++()
    {
        k++; i = k / n; j = k % n;
    }

    template <typename T>
    bool  operator!=(T dummy)
    {
        return !(k == n*m);
    }

private:

    MatrixType *M;
    unsigned int i, j, k, n, m;
};
#endif // ALGEBRA_MATRIX
