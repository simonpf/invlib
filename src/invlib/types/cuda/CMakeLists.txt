cmake_minimum_required (VERSION 2.6)

include_directories("${PROJECT_SOURCE_DIR}/src/")

# Executables
if (CUDA_FOUND)
  include(FindCUDA)
  list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
  cuda_add_library(cuda_kernels kernels.cu)
endif (CUDA_FOUND)
