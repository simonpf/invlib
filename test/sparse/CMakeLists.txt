cmake_minimum_required (VERSION 2.6)

include_directories(SYSTEM "${Boost_INCLUDE_DIRS}"
                           "${EIGEN3_INCLUDE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/src/"
                    "${PROJECT_SOURCE_DIR}/test/")

link_directories(${Boost_LIBRARIES})

# Executables
if (Boost_FOUND)
  add_executable(sparse_data sparse_data.cpp)
endif (Boost_FOUND)