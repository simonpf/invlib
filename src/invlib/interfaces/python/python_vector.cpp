
template<typename Real>
PythonVector<Real>::PythonVector(Real   * other_data,
                                 size_t   rows,
                                 bool     copy)
    : n(rows), owner(copy)
{
    if (copy) {
        data = new Real[n];
        std::copy(other_data, other_data + n, data);
    } else {
        data = other_data;
    }
}

template<typename Real>
PythonVector<Real>::~PythonVector()
{
    if (owner) {
        delete data;
    }
}
