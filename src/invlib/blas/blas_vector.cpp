template
<
typename Real
>
BlasVector<Real>::BlasVector(const VectorData<Real> & v)
    : VectorData<Real>(v)
{
    // Nothing to do here.
}

template
<
typename Real
>
BlasVector<Real>::BlasVector(VectorData<Real> && v)
    : VectorData<Real>(v)
{
    // Nothing to do here.
}

template
<
typename Real
>
void BlasVector<Real>::accumulate(const BlasVector & v)
{
    blas::axpy(static_cast<int>(n),
         static_cast<Real>(1.0), v.get_element_pointer(), 1,
         *elements, 1);
}

template
<
typename Real
>
void BlasVector<Real>::accumulate(Real c)
{
    for (size_t i = 0; i < n; i++)
    {
        (*elements)[i] += c;
    }
}

template
<
typename Real
>
void BlasVector<Real>::subtract(const BlasVector & v)
{
    blas::axpy(static_cast<int>(n),
         static_cast<Real>(-1.0), v.get_element_pointer(), 1,
         *elements, 1);
}

template
<
typename Real
>
void BlasVector<Real>::scale(Real c)
{
    for (size_t i = 0; i < n; i++)
    {
        (*elements)[i] *= c;
    }
}

template
<
typename Real
>
Real BlasVector<Real>::norm() const
{
    return sqrt(dot(*this, *this));
}

template
<
typename Real
>
Real dot(const BlasVector<Real > & a, const BlasVector<Real> & b)
{
    return blas::dot<double>(
        static_cast<int>(a.rows()), a.get_element_pointer(), 1,
        b.get_element_pointer(), 1
        );
}
