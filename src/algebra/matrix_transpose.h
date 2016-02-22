#ifndef ALGEBRA_MATRIX_TRANSPOSE
#define ALGEBRA_MATRIX_TRANSPOSE

template
<
typename T1,
typename Matrix
>
class MatrixTranspose
{

public:

    using MatrixBase = Matrix;
    using Vector = typename Matrix::Vector;

    // ----------------- //
    //   Constructors    //
    // ----------------- //

    MatrixTranspose(const T1 &A_)
        : A(A_) {}

    // ------------------ //
    //      Addition      //
    // ------------------ //

    Matrix add(const Matrix& B)
    {
        Matrix C(static_cast<Matrix>(A).transpose_add(B));
        return C;
    }

    // ------------------ //
    //   Multiplication   //
    // ------------------ //

    template <typename T>
    T multiply(const T& B) const
    {

        Matrix C(static_cast<Matrix>(A).transpose_multiply(B));
        return C;
    }

    // ----------------- //
    // Addition Operator //
    // ----------------- //

    template <typename T>
    using Sum = MatrixSum<MatrixTranspose, T, Matrix>;

    template<typename T>
    Sum<T> operator+(const T &B)
    {
        return Sum<T>(*this, B);
    }

    // ----------------------- //
    // Multiplication Operator //
    // ----------------------- //

    template <typename T>
    using Product = MatrixProduct<MatrixTranspose, T, Matrix>;

    template<typename T>
    Product<T> operator*(const T &A) const
    {
        return Product<T>(*this, A);
    }

    Matrix invert() const
    {
        Matrix B(static_cast<Matrix>(A).transpose().invert());
        return B;
    }

    Vector solve(const Vector& v) const
    {
        Matrix B(static_cast<Matrix>(A).transpose().solve(v));
        return B;
    }

    operator Matrix() const
    {
        Matrix C(A.transpose());
        return C;
    }

private:

    typedef typename
        std::conditional<std::is_same<T1, Matrix>::value, const Matrix&, T1>::type A1;
    A1 A;

};

template
<
typename T
>
MatrixTranspose<T, typename T::MatrixBase> transp( const T &A )
{
    return MatrixTranspose<T, typename T::MatrixBase>( A );
}

#endif // ALGEBRA_MATRIX_TRANSPOSE
