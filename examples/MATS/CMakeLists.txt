cmake_minimum_required (VERSION 2.6)
project (invlib)


include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
include_directories("${PROJECT_SOURCE_DIR}/src/"
                    "${PROJECT_SOURCE_DIR}/test/")

# Executables
if (EIGEN3_FOUND)
    add_executable(oem oem.cpp)
endif (EIGEN3_FOUND)
