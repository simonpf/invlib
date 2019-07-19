template
<
typename ScalarType
>
VectorData<ScalarType>::VectorData(const VectorData &v)
    : n(v.rows())
{
    elements = array::create<ScalarType>(n);
    if (n > 0) {
        std::copy(v.begin(), v.end(), elements.get());
    }
}

template
<
typename ScalarType
>
VectorData<ScalarType> & VectorData<ScalarType>::operator=(const VectorData &v)
{
    n = v.rows();
    elements  = array::create<ScalarType>(n);
    std::copy(v.begin(), v.end(), elements.get());
}

template
<
typename ScalarType
>
VectorData<ScalarType> VectorData<ScalarType>::random(size_t n)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> real_dis(-10,10);

    auto pointer = array::create<ScalarType>(n);
    for (size_t i = 0; i < n; i++) {
        pointer[i] = real_dis(gen);
    }
    return VectorData(n, pointer);
}

template
<
typename ScalarType
>
VectorData<ScalarType>::VectorData(size_t n_, std::shared_ptr<ScalarType[]> elements_)
    : n(n_), elements(elements_)
{
    // Nothing to do here.
}

template
<
typename ScalarType
>
VectorData<ScalarType>::VectorData(const VectorArchetype<ScalarType> &v)
    : n(v.rows())
{
    elements = array::create<ScalarType>(n);
    for (size_t i = 0; i < n; i++) {
        (*elements)[i] = v(i);
    }
}

template
<
typename ScalarType
>
VectorData<ScalarType>::operator VectorArchetype<ScalarType>() const
{
    VectorArchetype<ScalarType> v; v.resize(n);
    for (size_t i = 0; i < n; i++)
    {
        v(i)= (*elements)[i];
    }
    return v;
}

template
<
typename ScalarType
>
void VectorData<ScalarType>::resize(size_t n_) {
    n = n_;
    elements = array::create<ScalarType>(n);
}

template
<
typename ScalarType
>
bool VectorData<ScalarType>::operator== (const VectorData & w) const
{
    bool result = true;

    const ScalarType * w_elements = w.get_element_pointer();
    for (size_t i = 0; i < n; i++)
    {
        result = result && numerical_equality(elements[i], w_elements[i]);
    }
    return result;
}

template <typename ScalarType>
std::ostream & operator<<(std::ostream & s, const VectorData<ScalarType> & vector)
{
    s << "Dense Vector Data:" << std::endl;
    s << "[";
    for (size_t i = 0; i < vector.n - 1; i++)
    {
        s << vector.elements[i] << " ";
    }
    s << vector.elements[vector.n - 1] << "]";
    s << std::endl;
    return s;
}
