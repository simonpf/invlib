#ifndef ALGEBRA_SUM
#define ALGEBRA_SUM

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
class MatrixSum
{

public:

    using Vector = typename Matrix::Vector;
    using MatrixBase = Matrix;

    MatrixSum( const T1 &Op1, const T2 &Op2 )
        : A(Op1), B(Op2) {}

    // ------------------ //
    //   Multiplication   //
    // ------------------ //

    Vector multiply(const Vector &v) const
    {
        Vector tmp = B.multiply(v);
        tmp += A.multiply(v);
        return tmp;
    }

    Matrix multiply(const Matrix &C) const
    {
        Matrix tmp1 = A.add(B);
        Matrix tmp2 = tmp1.multiply(C);
        return tmp2;
    }

    // ------------------ //
    //      Addition      //
    // ------------------ //

    Matrix add(const Matrix &C) const
    {
        Matrix tmp1 = A.add(B);
        Matrix tmp2 = tmp1.add(C);
        return tmp2;
    }

    template <typename T3>
    using Product = MatrixProduct<MatrixSum, T3, Matrix>;

    template<typename T3>
    auto operator*(const T3& C) const -> Product<T3>
    {
        return Product<T3>((*this), C);
    }

    // ------------------ //
    //  Nested  Addition  //
    // ------------------ //

    template <typename T3>
    using Sum = MatrixSum<T3, MatrixSum, Matrix>;

    template <typename T3>
    auto operator+(const T3 &C) const -> Sum<T3> const
    {
        return Sum<T3>(C, *this);
    }

    template<typename T>
    operator T() const
    {
        T tmp1 = B;
        T tmp2 = A.add(tmp1);
        return tmp2;
    }

private:

    // Operand references.
    typedef typename
        std::conditional<std::is_same<T1, Vector>::value, const Vector&, T1>::type A1;

    typedef typename
        std::conditional<std::is_same<T1, Matrix>::value, const Matrix&, T1>::type A2;

    typedef typename
        std::conditional<std::is_same<T2, Vector>::value, const Vector&, T2>::type B1;

    typedef typename
        std::conditional<std::is_same<T2, Matrix>::value, const Matrix&, T2>::type B2;

    A2 A;
    B2 B;

};

#endif // ALGEBRA_SUM
