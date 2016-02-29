cmake_minimum_required (VERSION 2.6)

add_subdirectory(Eigen)
add_subdirectory(optimization)
add_subdirectory(algebra)
add_subdirectory(forward_models)

SET(CMAKE_CXX_FLAGS "-O3")

include_directories ("${PROJECT_SOURCE_DIR}/src/"
                     "${PROJECT_SOURCE_DIR}/test/"
                     "/home/simon/Build/boost_1_60_0/")
link_directories("/home/simon/Build/boost_1_60_0/stage/lib")
