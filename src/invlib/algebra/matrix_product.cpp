// ---------------------  //
//  Matrix Product Class  //
// ---------------------  //

template
<
typename T1,
typename T2
>
template <typename T3>
auto MatrixProduct<T1, T2>::multiply(const T3 &t) const
    -> typename T3::ResultType
{
    using T3ResultType = typename T3::ResultType;

    T3ResultType u = remove_reference_wrapper(B).multiply(t);
    T3ResultType v = remove_reference_wrapper(A).multiply(u);
    return v;
}

template
<
typename T1,
typename T2
>
auto MatrixProduct<T1, T2>::invert() const
    -> MatrixType
{
    MatrixType D = this->operator ResultType();
    MatrixType E = D.invert();
    return E;
}

template
<
typename T1,
typename T2
>
auto MatrixProduct<T1, T2>::solve(const VectorType &u) const
    -> VectorType
{
    VectorType v = B.solve(u);
    VectorType w = A.solve(v);
    return w;
}

template
<
typename T1,
typename T2
>
auto MatrixProduct<T1, T2>::transpose() const
    -> MatrixType
{
    MatrixType C = A.multiply((MatrixType) B);
    MatrixType D = C.transpose();
    return D;
}

template
<
typename T1,
typename T2
>
MatrixProduct<T1, T2>::operator ResultType() const
{
    ResultType C = A.multiply(static_cast<const ResultType &>(B));
    return C;
}
