template <typename Base>
Vector<Base>::Vector(Base &&b)
    : Base(std::forward<Base>(v)) {}

template <typename Base>
void Vector<Base>::accumulate(const Vector &v);
{
    this->operator+=(B);
}

template <typename T>
    template <typename Base>
auto Vector<Base>::operator+(T &&v) const
    -> Sum<T>
{
    return Sum<T>(*this, std::forward<T>(v));
}

template <typename T>
    template<typename Base>
auto Vector<Base>::operator-(T &&v) const -> Difference<T>
{
    return Difference<T>(*this, std::forward<T>(v));
}

template<typename Base>
auto dot(const Vector<Base>& v, const Vector<Base>& w) -> decltype(v.dot(w))
{
    return v.dot(w);
}
