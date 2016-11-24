CudaDevice::CudaDevice()
    : cuda_allocator()
{
    cublasCreate(&cublas_handle);
    cusparseCreate(&cusparse_handle);
}

dim3 CudaDevice::get_1d_grid(int m) const
{
    int grid_size = m / block_size;
    if ((m % block_size) != 0) {
        grid_size += 1;
    }
    return dim3(grid_size);
}

dim3 CudaDevice::get_1d_block() const
{
    return dim3(block_size);
}

CudaDevice Cuda::default_device{};
