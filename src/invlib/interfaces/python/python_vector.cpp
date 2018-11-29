
// --------------------------  //
//     Python Vector Data      //
// --------------------------  //

template<typename Real>
PythonVectorData<Real>::PythonVectorData(Real   * other_elements,
                                         size_t   rows,
                                         bool     copy)
    : n(rows), owner(copy)
{
    if (copy) {
        elements = new Real[n];
        std::copy(other_elements, other_elements + n, elements);
    } else {
        elements = other_elements;
    }
}

template<typename Real>
PythonVectorData<Real>::PythonVectorData(const PythonVectorData<Real> &v)
    : n(v.n), owner(true)
{
    elements = new Real[n];
    std::copy(v.elements, v.elements + n, elements);
}

template<typename Real>
PythonVectorData<Real>::PythonVectorData(PythonVectorData<Real> &&v)
    : n(v.n), owner(true), elements(v.elements)
{
    v.owner = false;
}

template<typename Real>
PythonVectorData<Real> & PythonVectorData<Real>::operator=(const PythonVectorData<Real> &v)
{
    if (elements) {
        delete elements;
    }

    n     = v.n;
    owner = true;
    elements = new Real[n];
    std::copy(v.elements, v.elements + n, elements);
}

template<typename Real>
PythonVectorData<Real> & PythonVectorData<Real>::operator=(PythonVectorData<Real> &&v)
{
    if (elements) {
        delete elements;
    }

    n     = v.n;
    owner = true;
    elements = v.elements;
    v.owner = false;
}

template<typename Real>
PythonVectorData<Real>::~PythonVectorData()
{
    if (owner) {
        delete elements;
    }
}

template<typename Real>
PythonVectorData<Real> PythonVectorData<Real>::get_block(size_t i,
                                                         size_t di) const
{
    return PythonVectorData<Real>(elements + i, di, false);
}


// --------------------------  //
//       Python Vector         //
// --------------------------  //

template<typename Real>
PythonVector<Real>::PythonVector(Real   * other_elements,
                                 size_t   rows,
                                 bool     copy)
{
    PythonVectorData<Real>::operator=(PythonVectorData<Real>(other_elements, rows, copy));
}
