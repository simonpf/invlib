cmake_minimum_required (VERSION 2.6)

include_directories("${PROJECT_SOURCE_DIR}/src/"
                    "${PROJECT_SOURCE_DIR}/test/")

link_directories(${Boost_LIBRARIES}
                "/home/simon/src/arts/build/src")

# Executables
if (Boost_FOUND)

  include_directories(SYSTEM "${Boost_INCLUDE_DIRS}")

  add_executable(sphere sphere.cpp)
  target_link_libraries(sphere ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
  add_executable(linear linear.cpp)
  target_link_libraries(linear ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif (Boost_FOUND)
