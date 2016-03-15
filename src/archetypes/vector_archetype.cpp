template
<
typename Real
>
VectorArchetype<Real>::VectorArchetype(const VectorArchetype<Real> &v)
    : n(v.rows())
{
    delete data;
    data = new Real[n];
    std::copy(std::begin(v.data), std::end(v.data), std::begin(data));
}

template
<
typename Real
>
VectorArchetype<Real>& VectorArchetype<Real>::operator=(const VectorArchetype<Real> &v)
{
    n = v.rows();
    delete data;
    data = new Real[n];
    std::copy(std::begin(v.data), std::end(v.data), std::begin(data));
}

template
<
typename Real
>
void VectorArchetype<Real>::resize(unsigned int i)
{
    n = i;
    delete data;
    data = new Real[n];
}

template
<
typename Real
>
Real & VectorArchetype<Real>::operator()(unsigned int i)
{
    return data[i];
}

template
<
typename Real
>
Real VectorArchetype<Real>::operator()(unsigned int i) const
{
    return data[i];
}


template
<
typename Real
>
unsigned int VectorArchetype<Real>::rows() const
{
    return n;
}

template
<
typename Real
>
void VectorArchetype<Real>::accumulate(const VectorArchetype<Real> &v)
{
    assert(n == v.rows());

    for (unsigned int i = 0; i < n; i++)
    {
	(*this)(i) += v(i);
    }
}

template
<
typename Real
>
void VectorArchetype<Real>::subtract(const VectorArchetype<Real>& v)
{
    assert(n == v.rows());

    for (unsigned int i = 0; i < n; i++)
    {
	(*this)(i) -= v(i);
    }
}

template
<
typename Real
>
void VectorArchetype<Real>::scale(Real c)
{
    for (unsigned int i = 0; i < n; i++)
    {
	(*this)(i) *= c;
    }
}

template
<
typename Real
>
Real norm(const VectorArchetype<Real> &v)
{
    return sqrt(dot(v, v));
}

template
<
typename Real
>
Real dot(const VectorArchetype<Real> &v, const VectorArchetype<Real> &w)
{
    Real sum = 0.0;
    for (unsigned int i = 0; i < v.rows(); i++)
    {
	sum += v(i) * w(i);
    }
    return sum;
}
    
template
<
typename Real
>
std::ostream & operator<<(std::ostream &out, const VectorArchetype<Real>& v)
{
    out << "[";
    for (unsigned int i = 0; i < v.rows()-1; i++)
    {
	out << v(i) << ", ";
    }
    out << v(v.rows() - 1) << "] " << std::endl;
}
	
