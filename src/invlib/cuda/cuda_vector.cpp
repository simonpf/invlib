#define HANDLE_CUDA_ERROR(x) {handle_cuda_error((x), __FILE__, __LINE__);}

template <typename Real>
CudaVector<Real>::CudaVector(CudaDevice & device_)
    : n(0), device(&device_), allocator(&device->get_allocator()),
    deleter(&device->get_allocator())
{
    // Nothing to do here.
}

template <typename Real>
CudaVector<Real>::CudaVector(const CudaVector & v)
    : n(v.n), device(v.device), allocator(v.allocator),
    deleter(&v.device->get_allocator())
{
    if (n > 0)
    {
        elements  = std::shared_ptr<Real *>(new (Real *), deleter);
        *elements = static_cast<Real *>(allocator->request(n * sizeof(Real)));

        const Real * vector_elements = v.get_element_pointer();
        cudaError_t error = cudaMemcpy(reinterpret_cast<      void *>(*elements),
                                       reinterpret_cast<const void *>(vector_elements),
                                       n * sizeof(Real),
                                       cudaMemcpyDeviceToDevice);
        HANDLE_CUDA_ERROR(error);
    }
}

template <typename Real>
CudaVector<Real>::CudaVector(const VectorData<Real> & vector,
                             CudaDevice & device_)
    : n(vector.rows()), device(&device_), allocator(&device->get_allocator()),
    deleter(&device->get_allocator())
{
    if (n > 0)
    {
    elements  = std::shared_ptr<Real *>(new (Real *), deleter);
    *elements = static_cast<Real *>(allocator->request(n * sizeof(Real)));

    const Real * vector_elements = vector.get_element_pointer();
    cudaError_t error = cudaMemcpy(reinterpret_cast<      void *>(*elements),
                                   reinterpret_cast<const void *>(vector_elements),
                                   n * sizeof(Real),
                                   cudaMemcpyHostToDevice);
    HANDLE_CUDA_ERROR(error);
    }
}

template <typename Real>
CudaVector<Real>::CudaVector(VectorData<Real> && vector,
                             CudaDevice & device_)
    : n(vector.rows()), device(&device_), allocator(&device->get_allocator()),
    deleter(&device->get_allocator())
{
    elements  = std::shared_ptr<Real *>(new (Real *), deleter);
    *elements = static_cast<Real *>(allocator->request(n * sizeof(Real)));

    const Real * vector_elements = vector.get_element_pointer();
    cudaError_t error = cudaMemcpy(reinterpret_cast<      void *>(*elements),
                                   reinterpret_cast<const void *>(vector_elements),
                                   n * sizeof(Real),
                                   cudaMemcpyHostToDevice);
    HANDLE_CUDA_ERROR(error);
}

template <typename Real>
void CudaVector<Real>::resize(size_t n_)
{
    n = n_;
    elements  = std::shared_ptr<Real *>(new (Real *), deleter);
    *elements = static_cast<Real *>(allocator->request(n * sizeof(Real)));
}

template <typename Real>
void CudaVector<Real>::accumulate(const CudaVector & v)
{
    cublas::axpy(device->get_cublas_handle(), n, static_cast<Real>(1.0),
                 v.get_element_pointer(), 1, *elements, 1);
}

template <typename Real>
void CudaVector<Real>::accumulate(RealType c)
{
    cuda::accumulate(*elements, c, n,
                     device->get_1d_grid(n),
                     device->get_1d_block());
}

template <typename Real>
void CudaVector<Real>::subtract(const CudaVector & v)
{
    cublas::axpy(device->get_cublas_handle(), n, static_cast<Real>(-1.0),
                 v.get_element_pointer(), 1, *elements, 1);
}

template <typename Real>
void CudaVector<Real>::scale(RealType c)
{
    cuda::scale(*elements, c, n, device->get_1d_grid(n), device->get_1d_block());
}

template <typename Real>
CudaVector<Real>::operator VectorData<Real>() const
{
    VectorData<Real> v{};
    v.resize(n);
    Real * vector_elements = v.get_element_pointer();

    cudaError_t error = cudaMemcpy(static_cast<      void *>(vector_elements),
                                   static_cast<const void *>(*elements),
                                   n * sizeof(Real),
                                   cudaMemcpyDeviceToHost);
    HANDLE_CUDA_ERROR(error);
    return v;
}

template <typename Real>
Real CudaVector<Real>::norm() const
{
    return sqrt(dot(*this, *this));
}

template
<
typename RealType
>
RealType dot(const CudaVector<RealType>& v, const CudaVector<RealType>& w)
{
    int n = v.rows();
    return cublas::dot(v.device->get_cublas_handle(), n,
                       v.get_element_pointer(), 1,
                       w.get_element_pointer(), 1);
}
