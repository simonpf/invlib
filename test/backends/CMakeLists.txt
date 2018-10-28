cmake_minimum_required (VERSION 2.6)

include_directories("${PROJECT_SOURCE_DIR}/src/"
                    "${PROJECT_SOURCE_DIR}/test/")



if (CUDA_FOUND)
  include(FindCUDA)
  add_definitions(-DCUDA)
endif(CUDA_FOUND)

if (MKL_FOUND)
  include(FindMKL)
  add_definitions(-DMKL)
endif(MKL_FOUND)

# Executables
if (CUDA_FOUND AND Boost_FOUND)

  include_directories(SYSTEM "${Boost_INCLUDE_DIRS}")

  link_directories(${Boost_LIBRARIES})

  cuda_add_executable(backends backends.cpp)
  target_link_libraries(backends cuda_kernels
                        ${MKL_LIBRARIES}
                        ${CUDA_CUBLAS_LIBRARIES}
                        ${CUDA_cusparse_LIBRARY})
endif (CUDA_FOUND AND Boost_FOUND)
