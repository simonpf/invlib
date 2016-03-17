template <typename Matrix>
MatrixIdentity<Matrix>::MatrixIdentity() : c(1.0)
{
    // Nothing to do here.
}

template <typename Matrix>
MatrixIdentity<Matrix>::MatrixIdentity(RealType c_) : c(c_)
{
    // Nothing to do here.
}

template <typename Matrix>
auto MatrixIdentity<Matrix>::scale() const
    ->  RealType
{
    return c;
}

template <typename Matrix>
    template <typename T1>
auto MatrixIdentity<Matrix>::multiply(T1 &&B) const
    -> T1 &&
{
    B.scale(c);
    return std::forward<T1>(B);
}

template <typename Matrix>
    template <typename T1>
auto MatrixIdentity<Matrix>::solve(T1 &&B) const
    -> T1 &&
{
    B.scale(1.0 / c);
    return std::forward<T1>(B);
}

template <typename Matrix>
    template<typename T1>
auto MatrixIdentity<Matrix>::operator*(T1 &&A) const
    -> Product<T1>
{
    return Product<T1>(*this, A);
}
