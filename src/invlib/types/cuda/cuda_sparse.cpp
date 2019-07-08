#define HANDLE_CUDA_ERROR(x) {handle_cuda_error((x), __FILE__, __LINE__);}

template <typename Real>
cusparseMatDescr CudaSparse<Real, Representation::CompressedColumns>::
default_matrix_descriptor = {
    MatrixType : CUSPARSE_MATRIX_TYPE_GENERAL,
    FillMode   : CUSPARSE_FILL_MODE_LOWER,
    DiagType   : CUSPARSE_DIAG_TYPE_NON_UNIT,
    IndexBase  : CUSPARSE_INDEX_BASE_ZERO};

template<typename Real>
CudaSparse<Real, Representation::CompressedColumns>::CudaSparse(
    const SparseData<Real, int, Representation::CompressedColumns> & base,
    CudaDevice & device_)
    : m(base.rows()), n(base.cols()), nnz(base.non_zeros()), device(device_)
{
    const int * row_indices_host = base.get_row_index_pointer();
    row_indices = std::shared_ptr<int *>(new (int *), int_deleter);
    HANDLE_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&*row_indices),
                                 nnz * sizeof(int)));
    HANDLE_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void *>(*row_indices),
                                 reinterpret_cast<const void *>(row_indices_host),
                                 nnz * sizeof(int),
                                 cudaMemcpyHostToDevice));

    const int * column_starts_host = base.get_column_start_pointer();
    column_starts = std::shared_ptr<int *>(new (int *), int_deleter);
    HANDLE_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&*column_starts),
                                 (n + 1) * sizeof(int)));
    HANDLE_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void *>(*column_starts),
                                 reinterpret_cast<const void *>(column_starts_host),
                                 (n + 1) * sizeof(int),
                                 cudaMemcpyHostToDevice));

    elements = std::shared_ptr<Real *>(new (Real *), real_deleter);
    HANDLE_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&*elements),
                                 nnz * sizeof(Real)));
    HANDLE_CUDA_ERROR(cudaMemcpy(
                          reinterpret_cast<void *>(*elements),
                          reinterpret_cast<const void *>(base.get_element_pointer()),
                          nnz * sizeof(Real),
                          cudaMemcpyHostToDevice));
}

template<typename Real>
auto CudaSparse<Real, Representation::CompressedColumns>::multiply(
    const VectorType & v) const
    -> VectorType
{
    VectorType w{};
    w.resize(m);
    cusparse::smv<Real, Representation::CompressedColumns>(
        device.get_cusparse_handle(),
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        m, n, nnz, 1.0,
        reinterpret_cast<cusparseMatDescr_t>(&default_matrix_descriptor),
        *elements, *column_starts, *row_indices,
        v.get_element_pointer(), 0.0,
        w.get_element_pointer());
    return w;
}

template<typename Real>
auto CudaSparse<Real, Representation::CompressedColumns>::transpose_multiply(
    const VectorType & v) const
    -> VectorType
{
    VectorType w{};
    w.resize(n);
    cusparse::smv<Real, Representation::CompressedColumns>(
        device.get_cusparse_handle(),
        CUSPARSE_OPERATION_TRANSPOSE,
        m, n, nnz, 1.0,
        reinterpret_cast<cusparseMatDescr_t>(&default_matrix_descriptor),
        *elements, *column_starts, *row_indices,
        v.get_element_pointer(), 0.0,
        w.get_element_pointer());
    return w;
}

template <typename Real>
cusparseMatDescr CudaSparse<Real, Representation::CompressedRows>::
default_matrix_descriptor = {
    MatrixType : CUSPARSE_MATRIX_TYPE_GENERAL,
    FillMode   : CUSPARSE_FILL_MODE_LOWER,
    DiagType   : CUSPARSE_DIAG_TYPE_NON_UNIT,
    IndexBase  : CUSPARSE_INDEX_BASE_ZERO};

template<typename Real>
CudaSparse<Real, Representation::CompressedRows>::CudaSparse(
    const SparseData<Real, int, Representation::CompressedRows> & base,
    CudaDevice & device_)
    : m(base.rows()), n(base.cols()), nnz(base.non_zeros()), device(device_)
{
    const int * column_indices_host = base.get_column_index_pointer();
    column_indices = std::shared_ptr<int *>(new (int *), int_deleter);
    HANDLE_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&*column_indices),
                                 nnz * sizeof(int)));
    HANDLE_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void *>(*column_indices),
                                 reinterpret_cast<const void *>(column_indices_host),
                                 nnz * sizeof(int),
                                 cudaMemcpyHostToDevice));

    const int * row_starts_host = base.get_row_start_pointer();
    row_starts = std::shared_ptr<int *>(new (int *), int_deleter);
    HANDLE_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&*row_starts),
                                 (m + 1) * sizeof(int)));
    HANDLE_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void *>(*row_starts),
                                 reinterpret_cast<const void *>(row_starts_host),
                                 (m + 1) * sizeof(int),
                                 cudaMemcpyHostToDevice));

    elements = std::shared_ptr<Real *>(new (Real *), real_deleter);
    HANDLE_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&*elements),
                                 nnz * sizeof(Real)));
    HANDLE_CUDA_ERROR(cudaMemcpy(
                          reinterpret_cast<void *>(*elements),
                          reinterpret_cast<const void *>(base.get_element_pointer()),
                          nnz * sizeof(Real),
                          cudaMemcpyHostToDevice));
}

template<typename Real>
auto CudaSparse<Real, Representation::CompressedRows>::multiply(
    const VectorType & v) const
    -> VectorType
{
    VectorType w{};
    w.resize(m);

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(& descr);
    cusparse::smv<Real, Representation::CompressedRows>(
        device.get_cusparse_handle(),
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        m, n, nnz, 1.0,
        reinterpret_cast<cusparseMatDescr_t>(&default_matrix_descriptor),
        *elements, *row_starts, *column_indices,
        v.get_element_pointer(), 0.0,
        w.get_element_pointer());
    return w;
}

template<typename Real>
auto CudaSparse<Real, Representation::CompressedRows>::transpose_multiply(
    const VectorType & v) const
    -> VectorType
{
    VectorType w{};
    w.resize(n);
    cusparse::smv<Real, Representation::CompressedRows>(
        device.get_cusparse_handle(),
        CUSPARSE_OPERATION_TRANSPOSE,
        m, n, nnz, 1.0,
        reinterpret_cast<const cusparseMatDescr_t>(&default_matrix_descriptor),
        *elements, *row_starts, *column_indices,
        v.get_element_pointer(), 0.0,
        w.get_element_pointer());
    return w;
}

template<typename Real>
CudaSparse<Real, Representation::Hybrid>::CudaSparse(
    const SparseData<Real, int, Representation::Coordinates> & base,
    CudaDevice & device_)
    : CSCBase(base), CSRBase(base)
{
    // Nothing to do here.
}

template<typename Real>
auto CudaSparse<Real, Representation::Hybrid>::multiply(
    const VectorType & v) const
    -> VectorType
{
    VectorType w{};
    w.resize(m);

    cusparse::smv<Real, Representation::CompressedRows>(
        device.get_cusparse_handle(),
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        m, n, nnz, 1.0,
        reinterpret_cast<cusparseMatDescr_t>(&default_matrix_descriptor),
        *CSRBase::elements, *row_starts, *column_indices,
        v.get_element_pointer(), 0.0,
        w.get_element_pointer());
    return w;
}

template<typename Real>
auto CudaSparse<Real, Representation::Hybrid>::transpose_multiply(
    const VectorType & v) const
    -> VectorType
{
    VectorType w{};
    w.resize(n);
    cusparse::smv<Real, Representation::CompressedRows>(
        device.get_cusparse_handle(),
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        n, m, nnz, 1.0,
        reinterpret_cast<const cusparseMatDescr_t>(&default_matrix_descriptor),
        *CSCBase::elements, *column_starts, *row_indices,
        v.get_element_pointer(), 0.0,
        w.get_element_pointer());
    return w;
}
