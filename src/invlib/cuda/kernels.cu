#include "kernels.h"

namespace invlib {
namespace cuda {

template<typename Real>
__global__ void accumulate_kernel(Real * x, Real c, int m)
{
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   if (i < m)
   {
       x[i] += c;
   }
}

template<typename Real>
__global__ void scale_kernel(Real * x, Real c, int m)
{
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   if (i < m)
   {
       x[i] *= c;
   }
}

template<typename Real>
void accumulate(Real * x, Real c, int m, dim3 grid, dim3 block)
{
    accumulate_kernel<<<grid, block>>>(x, c, m);
}

template<typename Real>
void scale(Real * x, Real c, int m, dim3 grid, dim3 block)
{
    scale_kernel<<<grid, block>>>(x, c, m);
}

template void accumulate<float>(float * x, float c, int m, dim3 grid, dim3 block);
template void accumulate<double>(double * x, double c, int m, dim3 grid, dim3 block);
template void scale<float>(float * x, float c, int m, dim3 grid, dim3 block);
template void scale<double>(double * x, double c, int m, dim3 grid, dim3 block);

} // namespace cuda
} // namespace invlib

