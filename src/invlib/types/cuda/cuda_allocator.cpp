#define HANDLE_CUDA_ERROR(x) {handle_cuda_error((x), __FILE__, __LINE__);}

CudaAllocator::~CudaAllocator()
{
    for (auto p : pointers)
    {
        cudaFree(p);
   }
}

size_t CudaAllocator::find_vector(size_t bytes)
{
    size_t i = 0;
    for (i = 0; i < sizes.size(); i++)
    {
        if ((bytes <= sizes[i]) && available[i])
        {
            available[i] = false;
            break;
        }
    }

    return i;
}

void * CudaAllocator::request(size_t bytes)
{
    size_t index = find_vector(bytes);

    if (!(index < sizes.size()))
    {
        sizes.push_back(bytes);
        pointers.push_back(nullptr);
        available.push_back(false);
        HANDLE_CUDA_ERROR(cudaMalloc(&pointers.back(), bytes));
    }

    return pointers[index];
}

void CudaAllocator::release(void * pointer)
{
    size_t i;
    for (i = 0; i < pointers.size(); i++)
    {
        if (pointers[i] == pointer)
        {
            available[i] = true;
            break;
        }
    }
    assert(i < pointers.size());
}
