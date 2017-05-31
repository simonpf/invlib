cmake_minimum_required (VERSION 2.6)

include_directories("${PROJECT_SOURCE_DIR}/src/"
                    "${PROJECT_SOURCE_DIR}/test/")

# Executables
if (Boost_FOUND AND EIGEN3_FOUND)

  include_directories(SYSTEM "${EIGEN3_INCLUDE_DIR}")
  include_directories(SYSTEM "${Boost_INCLUDE_DIRS}")

  link_directories(${Boost_LIBRARIES})

  add_executable(sparse_data sparse_data.cpp)

endif (Boost_FOUND AND EIGEN3_FOUND)
