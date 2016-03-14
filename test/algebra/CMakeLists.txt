cmake_minimum_required (VERSION 2.6)
project (invlib)

# Include directories.
include_directories ("${PROJECT_SOURCE_DIR}/src/"
                     "${PROJECT_SOURCE_DIR}/test/"
                     "/home/simon/Build/boost_1_60_0/"
                     "/home/simon/Documents/arts/arts_clean/src"
                     "/home/simon/Documents/arts/arts_clean/build/src")

# Link directories.
link_directories("/home/simon/Build/boost_1_60_0/stage/lib"
                 "/home/simon/Documents/arts/arts_clean/build/src"
                 "/home/simon/Build/OpenBLAS/")

SET(CMAKE_CXX_FLAGS "-O3")

# Executables
add_executable(identities identities.cc)
target_link_libraries(identities matpack openblas)

add_executable(solvers solvers.cc)
target_link_libraries(solvers matpack openblas)

add_executable(precision_matrix precision_matrix.cc)
target_link_libraries(precision_matrix matpack openblas)

add_executable(transformation transformation.cc)
target_link_libraries(transformation matpack openblas)

add_executable(invlib lazy_evaluation.cc)
target_link_libraries(invlib matpack openblas)
