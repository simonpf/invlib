#ifndef BENCHMARK_BENCHMARK_TYPES_H
#define BENCHMARK_BENCHMARK_TYPES_H

#include <utility>
#include "invlib/sparse/sparse_data.h"
#include "utility.h"

#ifdef CUDA
#include "invlib/cuda/cuda_sparse.h"
#endif

#ifdef MKL
#include "invlib/mkl/mkl_sparse.h"
#endif

namespace invlib {

#ifdef CUDA
using CudaSparseCR    = CudaSparse<double, Representation::CompressedRows>;
using CudaSparseCC    = CudaSparse<double, Representation::CompressedColumns>;
using CudaSparseHY    = CudaSparse<double, Representation::CompressedColumns>;
using CudaSparseTypes = std::tuple<CudaSparseCR, CudaSparseCC, CudaSparseHY>;
#else
using CudaSparseTypes = std::tuple<>;
#endif // CUDA

#ifdef MKL
using MklSparseCO    = MklSparse<double, Representation::Coordinates>;
using MklSparseCR    = MklSparse<double, Representation::CompressedRows>;
using MklSparseCC    = MklSparse<double, Representation::CompressedColumns>;
using MklSparseHY    = MklSparse<double, Representation::Hybrid>;
using MklSparseTypes = std::tuple<MklSparseCO, MklSparseCR, MklSparseCC, MklSparseHY>;
#else
using MklSparseTypes = std::tuple<>;
#endif // MKL

using SparseTypes = typename ConcatTuple<CudaSparseTypes, MklSparseTypes>::Type;

}

#endif // BENCHMARK_BENCHMARK_TYPES_H
