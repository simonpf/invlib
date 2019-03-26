template
<
    typename Real,
    template <typename> typename VData
>
BlasVector<Real, VData>::BlasVector(const VData<Real> & v)
    : VData<Real>(v)
{
    // Nothing to do here.
}

template
<
    typename Real,
    template <typename> typename VData
>
BlasVector<Real, VData>::BlasVector(VData<Real> && v)
    : VData<Real>(std::forward<VData<Real>>(v))
{
    // Nothing to do here.
}

template
<
    typename Real,
    template <typename> typename VData
>
void BlasVector<Real, VData>::accumulate(const BlasVector & v)
{
    blas::axpy<Real>(n, 1.0,
                     v.get_element_pointer(), 1,
                     get_element_pointer(), 1);
}

template
<
    typename Real,
    template <typename> typename VData
>
void BlasVector<Real, VData>::accumulate(Real c)
{
    auto elements = get_element_pointer();
    for (size_t i = 0; i < n; i++)
    {
        elements[i] += c;
    }
}

template
<
    typename Real,
    template <typename> typename VData
>
void BlasVector<Real, VData>::subtract(const BlasVector & v)
{
    blas::axpy<Real>(n, -1.0,
                     v.get_element_pointer(), 1,
                     get_element_pointer(), 1);
}

template
<
    typename Real,
    template <typename> typename VData
>
void BlasVector<Real, VData>::scale(Real c)
{
    blas::axpy<Real>(n, c - 1.0,
                     get_element_pointer(), 1,
                     get_element_pointer(), 1);
}

template
<
    typename Real,
    template <typename> typename VData
>
Real BlasVector<Real, VData>::norm() const
{
    return sqrt(dot(*this, *this));
}

template
<
    typename Real,
    template <typename> typename VData
>
Real dot(const BlasVector<Real, VData> & a, const BlasVector<Real, VData> & b)
{
    return blas::dot<Real>(static_cast<int>(a.rows()),
                           a.get_element_pointer(), 1,
                           b.get_element_pointer(), 1);
}
