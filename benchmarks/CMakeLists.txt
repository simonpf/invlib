cmake_minimum_required (VERSION 2.6)

include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
include_directories("${PROJECT_SOURCE_DIR}/src/"
                    "${PROJECT_SOURCE_DIR}/benchmarks/"
                    "${PROJECT_SOURCE_DIR}/ext"
                    /opt/intel/mkl/include)

set (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Cuda Dependent Code
include(FindCUDA)
if    (CUDA_FOUND)
  add_definitions(-DCUDA)
  set(BACKEND_LIBRARIES ${BACKEND_LIBRARIES} ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusparse_LIBRARY} cuda_kernels)
endif (CUDA_FOUND)

# MLK Dependent Code
include(${PROJECT_SOURCE_DIR}/cmake/FindMKL.cmake)
if    (MKL_FOUND)
  add_definitions(-DMKL)
  set(BACKEND_LIBRARIES ${BACKEND_LIBRARIES} ${MKL_LIBRARIES}
    pthread dl ${MKL_LIBRARIES})
endif (MKL_FOUND)

# Executables
if(MATS_DATA)
  add_definitions(-DMATS_DATA=${MATS_DATA})
  add_executable(mats mats.cpp)
  target_link_libraries(mats ${BACKEND_LIBRARIES})
endif(MATS_DATA)

add_executable(matrix_vector_multiplication matrix_vector_multiplication.cpp)
target_link_libraries(matrix_vector_multiplication ${BACKEND_LIBRARIES})