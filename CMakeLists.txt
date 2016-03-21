
cmake_minimum_required (VERSION 2.6)
project (invlib)

set (CMAKE_CXX_STANDARD 11)

# Check for BOOST library.
find_package(Boost 1.60.0)
find_package(LAPACK)
find_package(BLAS)

include_directories ("${PROJECT_SOURCE_DIR}/src/"
                     "${PROJECT_SOURCE_DIR}/test/"
                     "${Boost_INCLUDE_DIRS}")

# Build tests.
add_subdirectory(test)

