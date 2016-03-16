// -------------- //
//  Vector Class  //
// -------------- //

template <typename Base>
Vector<Base>::Vector(Base &&v)
    : Base(std::forward<Base>(v))
{
    // Nothing to do here.
}

template <typename Base>
auto Vector<Base>::begin()
    -> ElementIterator
{
    return ElementIterator(this);
}

template <typename Base>
auto Vector<Base>::end()
    -> ElementIterator
{
    return ElementIterator(this, this->rows());
};

template <typename Base>
    template <typename T1>
auto Vector<Base>::operator+(T1 &&v) const
    -> Sum<T1>
{
    return Sum<T1>(*this, std::forward<T1>(v));
}

template<typename Base>
    template <typename T1>
auto Vector<Base>::operator-(T1 &&v) const -> Difference<T1>
{
    return Difference<T1>(*this, std::forward<T1>(v));
}

template<typename Base>
auto dot(const Vector<Base>& v, const Vector<Base>& w) -> decltype(v.dot(w))
{
    return v.dot(w);
}

// ------------------------------- //
//  Vector::ElementIterator Class  //
// ------------------------------- //

template<typename Base>
Vector<Base>::ElementIterator::ElementIterator(VectorType* v_)
    : v(v_), k(0), n(v_->rows())
{
    // Nothingn to do here.
}

template<typename Base>
Vector<Base>::ElementIterator::ElementIterator(VectorType* v_, unsigned int k_)
    : v(v_), k(k_), n(v_->rows())
{
    // Nothing to do here.
}

template<typename Base>
auto Vector<Base>::ElementIterator::operator*()
    -> RealType
{
    return v->operator()(k);
}

template<typename Base>
auto Vector<Base>::ElementIterator::operator++()
    -> RealType&
{
    k++;
}

template<typename Base>
bool Vector<Base>::ElementIterator::operator!=(ElementIterator it)
{
    return (k != it.k);
}
