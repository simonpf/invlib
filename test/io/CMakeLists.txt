cmake_minimum_required (VERSION 2.6)

include_directories("${PROJECT_SOURCE_DIR}/src/"
                    "${PROJECT_SOURCE_DIR}/test/"
                    "${PROJECT_SOURCE_DIR}/ext/")

link_directories(${Boost_LIBRARIES})

# Executables
if (Boost_FOUND AND EIGEN3_FOUND)

  include_directories(SYSTEM "${Boost_INCLUDE_DIRS}"
                           "${EIGEN3_INCLUDE_DIR}")

  add_executable(read_and_write read_and_write.cpp)

endif (Boost_FOUND AND EIGEN3_FOUND)
