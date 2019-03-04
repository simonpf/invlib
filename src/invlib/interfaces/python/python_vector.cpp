
// --------------------------  //
//       Python Vector         //
// --------------------------  //

template<typename Real>
PythonVector<Real>::PythonVector(Real   * other_elements,
                                 size_t   rows,
                                 bool     copy)
    : VectorData<Real>(rows, array::make_shared(other_elements, rows, copy))
{
    // Nothing to do here.
}
