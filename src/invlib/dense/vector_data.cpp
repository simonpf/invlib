template
<
typename Real
>
VectorData<Real>::VectorData(const VectorData &v)
    : n(v.rows())
{
    elements  = std::shared_ptr<Real *>(new (Real *), ArrayDeleter<Real *>());
    *elements = new Real[n];

    if (n > 0)
    {
        std::copy(v.begin(), v.end(), *elements);
    }
}

template
<
typename Real
>
VectorData<Real> & VectorData<Real>::operator=(const VectorData &v)
{
    n = v.rows();
    elements  = std::shared_ptr<Real *>(new (Real *), ArrayDeleter<Real *>());
    *elements = new Real[n];
    std::copy(v.begin(), v.end(), *elements);
}

template
<
typename Real
>
VectorData<Real> VectorData<Real>::random(size_t n)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> real_dis(-10,10);

    std::shared_ptr<Real *> pointer(new (Real *), ArrayDeleter<Real *>());
    *pointer = new Real[n];

    for (size_t i = 0; i < n; i++)
    {
        (*pointer)[i] = real_dis(gen);
    }

    return VectorData(n, pointer);
}

template
<
typename Real
>
VectorData<Real>::VectorData(size_t n_, std::shared_ptr<Real *> elements_)
    : n(n_), elements(elements_)
{
    // Nothing to do here.
}

template
<
typename Real
>
VectorData<Real>::VectorData(const VectorArchetype<Real> &v)
    : n(v.rows())
{
    elements  = std::shared_ptr<Real *>(new (Real *), ArrayDeleter<Real *>());
    *elements = new Real[n];

    for (size_t i = 0; i < n; i++)
    {
        (*elements)[i] = v(i);
    }
}

template
<
typename Real
>
VectorData<Real>::operator VectorArchetype<Real>() const
{
    VectorArchetype<Real> v; v.resize(n);
    for (size_t i = 0; i < n; i++)
    {
        v(i)= (*elements)[i];
    }
    return v;
}

template
<
typename Real
>
void VectorData<Real>::resize(size_t n_)
{
    n = n_;
    elements  = std::shared_ptr<Real *>(new (Real *), ArrayDeleter<Real *>());
    *elements = new Real[n];
}

template
<
typename Real
>
bool VectorData<Real>::operator== (const VectorData & w) const
{
    bool result = true;

    const Real * w_elements = w.get_element_pointer();
    for (size_t i = 0; i < n; i++)
    {
        result = result && numerical_equality((*elements)[i], w_elements[i]);
    }
    return result;
}

template <typename Real>
std::ostream & operator<<(std::ostream & s, const VectorData<Real> & vector)
{
    s << "Dense Vector Data:" << std::endl;
    s << "[";
    for (size_t i = 0; i < vector.n - 1; i++)
    {
        s << (*vector.elements)[i] << " ";
    }
    s << (*vector.elements)[vector.n - 1] << "]";
    s << std::endl;
    return s;
}
