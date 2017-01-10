cmake_minimum_required (VERSION 2.6)
project (invlib)

### Output build info.
message(STATUS "CMAKE_SYSTEM_NAME          = ${CMAKE_SYSTEM_NAME}"         )
message(STATUS "CMAKE_CXX_COMPILER_ID      = ${CMAKE_CXX_COMPILER_ID}"     )
message(STATUS "CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_COMPILER_IS_GNUCXX   = ${CMAKE_COMPILER_IS_GNUCXX}"  )

### C++11 support.
set (CMAKE_CXX_STANDARD 11)

### Libraries.
# Boost
find_package(Boost 1.60.0)
# BLAS
include(FindBLAS)
# LAPACK
include(FindLAPACK)
# MPI
include(FindMPI)
# CUDA
include(FindCUDA)
# Eigen
include(cmake/FindEigen3.cmake)
# MKL
include(cmake/FindMKL.cmake)

set (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

if (NOT EIGEN3_FOUND)
  message(WARNING "Could not find Eigen3 library, which is required to build some \
                   examples. If Eigen3 is not detected automatically consider \
                   setting EIGEN3_INCLUDE_DIR to the path containing Eigen3.")
endif (NOT EIGEN3_FOUND)


include_directories ("${PROJECT_SOURCE_DIR}/src/"
                     "${PROJECT_SOURCE_DIR}/test/"
                     "${Boost_INCLUDE_DIRS}")

# Build tests.
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(examples)
add_subdirectory(benchmarks)
add_subdirectory(retrievals)

