/**
 * \file cuda/kernels.h
 *
 * \brief Header files for CUDA kernels, that are
 * compiled seperately.
 *
 */

#ifndef CUDA_KERNELS_H
#define CUDA_KERNERLS_H

#include "cuda.h"

namespace invlib {
namespace cuda {

/*! Add a constant to the array x.
 *
 * Launches an accumulation kernel on the device using
 * the \p grid and \p block dim3 objects for the block grid
 * and the thread grid, respectively. Both grids are assumed
 * to be 1-dimensional and gridDimX * blockDimX >= m.
 *
 * \param x Pointer to the CUDA device array.
 * \param c The constant to add to each element of the array.
 * \param n Number of elements of the array.
 * \param grid dim3 object representing a 1D grid of CUDA blocks.
 * \param block dim3 object representing a 1D grid ob CUDA threads.
 */
template<typename Real>
void accumulate(Real * x, Real c, int n, dim3 grid, dim3 block);

/*! Scale array by a constant.
 *
 * Launches a scaling kernel on the device using
 * the \p grid and \p block dim3 objects for the block grid
 * and the thread grid, respectively. Both grids are assumed
 * to be 1-dimensional and gridDimX * blockDimX >= m.
 *
 * \param x Pointer to the CUDA device array.
 * \param c The constant to scale each element of the array with.
 * \param n Number of elements of the array.
 * \param grid dim3 object representing a 1D grid of CUDA blocks.
 * \param block dim3 object representing a 1D grid ob CUDA threads.
 */
template<typename Real>
void scale(Real * x, Real c, int m, dim3 grid, dim3 block);

}      // namespace cuda
}      // namespace invlib
#endif // CUDA_KERNELS_H
