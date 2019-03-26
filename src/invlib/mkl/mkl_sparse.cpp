template
<
typename Real
>
MklSparse<Real, Representation::Coordinates>::MklSparse(
    const SparseData<Real, int, Representation::Coordinates> & matrix)
    : SparseData<Real, int, Representation::Coordinates>(matrix)
{
    // Nothing to do here.
}

template
<
typename Real
>
auto MklSparse<Real, Representation::Coordinates>::multiply(
    const VectorType & v) const
    -> VectorType
{
    VectorType w{}; w.resize(m);
    mkl::smv<Real, Representation::Coordinates>(
        'N', static_cast<int>(m), static_cast<int>(n), static_cast<int>(nnz), 1.0,
        get_element_pointer(),
        get_row_index_pointer(),
        get_column_index_pointer(),
        nullptr,
        v.get_element_pointer(), 0.0, w.get_element_pointer());
    return w;
}

template
<
typename Real
>
auto MklSparse<Real, Representation::Coordinates>::transpose_multiply(
    const VectorType & v) const
    -> VectorType
{
    VectorType w{}; w.resize(n);
    mkl::smv<Real, Representation::Coordinates>(
        'T', static_cast<int>(m), static_cast<int>(n), static_cast<int>(nnz),
        1.0, get_element_pointer(), get_row_index_pointer(), get_column_index_pointer(),
        nullptr, v.get_element_pointer(), 0.0, w.get_element_pointer());
    return w;
}

template
<
typename Real,
Representation rep
>
MklSparse<Real, rep>::MklSparse(const SparseData<Real, int, rep> & matrix)
    : SparseData<Real, int, rep>(matrix)
{
    mkl_matrix = mkl::sparse_create<Real, rep>(m, n,
                                               get_start_pointer(),
                                               get_index_pointer(),
                                               get_element_pointer());
}

template <typename Real, Representation rep>
template <typename T, typename TT>
auto MklSparse<Real, rep>::multiply(const T & v) const
    -> TT
{
    const auto & v_ = static_cast<TT>(v);
    TT w; w.resize(m);
    mkl::mv<Real, rep>(SPARSE_OPERATION_NON_TRANSPOSE,
                       static_cast<Real>(1.0), mkl_matrix, v_.get_element_pointer(),
                       static_cast<Real>(0.0), w.get_element_pointer());
    return w;
}

template <typename Real, Representation rep>
template <typename T>
auto MklSparse<Real, rep>::transpose_multiply(const T & v) const
    -> typename T::ResultType
{
    using ResultType = typename T::ResultType;
    const auto & v_ = static_cast<ResultType>(v);
    ResultType w; w.resize(n);
    mkl::mv<Real, rep>(SPARSE_OPERATION_TRANSPOSE,
                       static_cast<Real>(1.0), mkl_matrix, v_.get_element_pointer(),
                       static_cast<Real>(0.0), w.get_element_pointer());
    return w;
}

template
<
typename Real
>
MklSparse<Real, Representation::Hybrid>::MklSparse(
    const SparseData<Real, int, Representation::Coordinates> & matrix
    )
    : CSRBase(matrix), CSCBase(matrix)
{
    // Nothing to do here.
}

template
<
    typename Real
    >
MklSparse<Real, Representation::Hybrid>::MklSparse(
    const SparseData<Real, int, Representation::CompressedColumns> & matrix_csc,
    const SparseData<Real, int, Representation::CompressedRows> & matrix_csr
    )
    : CSCBase(matrix_csc), CSRBase(matrix_csr)
{
    // Nothing to do here.
}

