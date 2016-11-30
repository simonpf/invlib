cmake_minimum_required (VERSION 2.6)

include_directories("${PROJECT_SOURCE_DIR}/src/")

include(FindCUDA)

# Executables
if (CUDA_FOUND)
  list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
  cuda_add_library(cuda_kernels kernels.cu)
endif (CUDA_FOUND)