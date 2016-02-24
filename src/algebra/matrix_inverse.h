#ifndef ALGEBRA_MATRIX_INVERSE
#define ALGEBRA_MATRIX_INVERSE

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
class MatrixSum;

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

    MatrixInverse( const T1 &A_ )
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
        Vector w(A.solve(v));
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
        using Product = MatrixProduct<MatrixInverse, T, Matrix>;

    template <typename T>
    auto operator*(const T &B) const -> Product<T>
    {
        return Product<T>(*this, B);
    }

    // --------------------- //
    //   Addition  Operator  //
    // --------------------- //

    template<typename T>
    using Sum = MatrixSum<MatrixInverse, T, Matrix>;

    template <typename T>
    auto operator+(const T &B) const -> Sum<T>
    {
        return Sum<T>(*this, B);
    }

private:

    const T1 &A;

};

template
<
typename T
>
MatrixInverse<T, typename T::MatrixBase> inv( const T &A )
{
    return MatrixInverse<T, typename T::MatrixBase>( A );
}

#endif // ALGEBRA_MATRIX_INVERSE
