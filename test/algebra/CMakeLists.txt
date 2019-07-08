cmake_minimum_required (VERSION 2.6)

include_directories("${PROJECT_SOURCE_DIR}/src/"
                    "${PROJECT_SOURCE_DIR}/test/")

enable_testing()

add_executable(diagonal diagonal.cpp)
add_test(NAME diagonal COMMAND diagonal)

add_executable(identities identities.cpp)
add_test(NAME identities COMMAND diagonal)

add_executable(transformation transformation.cpp)
add_test(NAME transformation COMMAND transformation)

add_executable(solvers solvers.cpp)
add_test(NAME solvers COMMAND solvers)

add_executable(precision_matrix precision_matrix.cpp)
add_test(NAME precision_matrix COMMAND precision_matrix)
