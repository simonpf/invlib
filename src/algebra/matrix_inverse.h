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
        return A.invert();
    }

    template <typename Vector>
    Vector operator*(const Vector &v)
    {
        return A.solve(v);
    }

private:

    const Matrix &A;

}
