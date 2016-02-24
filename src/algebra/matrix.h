#ifndef ALGEBRA_MATRIX
#define ALGEBRA_MATRIX

#include "matrix_product.h"
#include "matrix_inverse.h"
#include "matrix_sum.h"

template
<
typename Base,
typename Vector_
>
class Matrix : public Base
{

public:

    using Vector = Vector_;
    using MatrixBase = Matrix;

    // ----------------- //
    // Base constructors //
    // ----------------- //

    /* template <typename ...Args> */
    /* Matrix(Args&... params) */
    /*     : Base(std::forward<Args>(params)...) {} */

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


    // ----------------- //
    // Addition operator //
    // ----------------- //

    Matrix add(const Matrix &B) const
    {
        Matrix C = static_cast<Matrix>(this->Base::operator+(B));
        return C;
    }

    template <typename T>
        using Sum = MatrixSum<Matrix, T, Matrix>;

    template<typename T>
    Sum<T> operator+(const T &B) const
    {
        return Sum<T>{B, *this};
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
    /* Sum operator+(const MatrixBase &B) */
    /* { */
    /*     return Sum(*this, B); */
    /* } */
};

#endif // ALGEBRA_MATRIX
