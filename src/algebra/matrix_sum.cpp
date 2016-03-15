template<typename T1, typename T2>
MatrixSum<T1, T2>::MatrixSum(T1 Op1, T2 Op2)
        : A(Op1), B(Op2)
{
    // Nothing to do here.
}

template<typename T1, typename T2>
auto MatrixSum<T1, T2>::multiply(const VectorType &v) const
    -> VectorType
{
    VectorType w = B.multiply(v);
    w.accumulate(A.multiply(v));
    return w;
}

template<typename T1, typename T2>
auto MatrixSum<T1, T2>::multiply(const MatrixType &C) const
    -> MatrixType
{
    MatrixType D = A;
    D.accum(B);
    return D.multiply(C);
}

template<typename T1, typename T2>
auto MatrixSum<T1, T2>::solve(const VectorType &v) const
    -> VectorType
{
    MatrixType C = A;
    A.accumulate(B);
    return A.solve(v);
}

template<typename T1, typename T2>
auto MatrixSum<T1, T2>::invert() const
    -> MatrixType
{
    MatrixType C = A;
    A.accumulate(B);
    return A.invert();
}

template<typename T1, typename T2>
    template<typename T3>
auto MatrixSum<T1, T2>::operator*(T3 &&C) const
    -> Product<T3>
{
    return Product<T3>(*this, C);
}

template<typename T1, typename T2>
    template <typename T3>
auto MatrixSum<T1, T2>::operator+(T3 &&C) const
    -> Sum<T3> const
{
    return Sum<T3>(*this, C);
}

template<typename T1, typename T2>
template <typename T3>
auto MatrixSum<T1, T2>::operator-(T3 &&C) const
    -> Difference<T3> const
{
    return Difference<T3>(*this, C);
}

template<typename T1, typename T2>
MatrixSum<T1, T2>::operator ResultType()
{
    ResultType C = A;
    C.accumulate(B);
    return C;
}
