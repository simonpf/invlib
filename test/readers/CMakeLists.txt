cmake_minimum_required (VERSION 2.6)

include_directories(SYSTEM "${Boost_INCLUDE_DIRS}"
                           "${EIGEN3_INCLUDE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/src/"
                    "${PROJECT_SOURCE_DIR}/test/"
                    "${PROJECT_SOURCE_DIR}/ext/pugixml")

link_directories(${Boost_LIBRARIES})

if (Boost_FOUND)
  add_executable(readers readers.cpp)
  add_executable(writers writers.cpp)
endif (Boost_FOUND)