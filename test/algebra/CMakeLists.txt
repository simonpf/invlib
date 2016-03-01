cmake_minimum_required (VERSION 2.6)
project (maplib)
include_directories ("${PROJECT_SOURCE_DIR}/src/"
                     "${PROJECT_SOURCE_DIR}/test/"
                     "/home/simon/Build/boost_1_60_0/")
link_directories("/home/simon/Build/boost_1_60_0/stage/lib")
SET(CMAKE_CXX_FLAGS "-O3")

add_executable(maplib lazy_evaluation.cc)
add_executable(identities identities.cc)
add_executable(transformation transformation.cc)
