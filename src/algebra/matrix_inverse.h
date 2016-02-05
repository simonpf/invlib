#ifndef ALGEBRA_MATRIX_INVERSE
#define ALGEBRA_MATRIX_INVERSE

template
<
typename Matrix
>
class MatrixInverse
{

public:

    MatrixInverse( const Matrix &A )
        : A_(A) {}

    operator Matrix()
    {
        return A_.invert();
    }

    template <typename Vector>
    Vector operator*(const Vector &v)
    {
        return A_.solve(v);
    }

    template <typename Vector>
    LazyProduct<MatrixInverse, Matrix> operator*(Matrix &B)
    {
        return LazyProduct<MatrixInverse, Matrix>(*this, B);
    }

private:

    const Matrix &A_;

};

template
<
typename Matrix
>
MatrixInverse<Matrix> inv( const Matrix &A )
{
    return MatrixInverse<Matrix>( A );
}

#endif // ALGEBRA_MATRIX_INVERSE
