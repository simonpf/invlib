cmake_minimum_required (VERSION 2.6)
project (invlib)

#
# Output build info.
#

message(STATUS "CMAKE_SYSTEM_NAME          = ${CMAKE_SYSTEM_NAME}"         )
message(STATUS "CMAKE_CXX_COMPILER_ID      = ${CMAKE_CXX_COMPILER_ID}"     )
message(STATUS "CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_COMPILER_IS_GNUCXX   = ${CMAKE_COMPILER_IS_GNUCXX}"  )

#
# C++17
#

set (CMAKE_CXX_STANDARD 17)

#
# Warnings
#

if(MSVC)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX")
else()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Werror")
endif()

#
# Libraries.
#

# Boost
find_package(Boost 1.60.0)
# BLAS
find_package(BLAS)
# LAPACK
find_package(LAPACK)
# MPI
include(FindMPI)
# CUDA
include(FindCUDA)
if (NOT WITH_CUDA)
  set(CUDA_FOUND "FALSE" CACHE STRING "Whether or not cuda was found.")
endif (NOT WITH_CUDA)
# Eigen
include(cmake/FindEigen3.cmake)
# MKL
include(cmake/FindMKL.cmake)

set (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Build tests.
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(examples)
add_subdirectory(benchmarks)
add_subdirectory(retrievals)

