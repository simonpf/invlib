cmake_minimum_required (VERSION 2.6)

add_subdirectory(algebra)
add_subdirectory(backends)
add_subdirectory(forward_models)
add_subdirectory(io)
add_subdirectory(mpi)
add_subdirectory(optimization)
add_subdirectory(profiling)
add_subdirectory(sparse)

include_directories(SYSTEM "${Boost_INCLUDE_DIRS}")
include_directories("${PROJECT_SOURCE_DIR}/src/"
                    "${PROJECT_SOURCE_DIR}/test/"
                    "${PROJECT_SOURCE_DIR}/ext/")

link_directories(${Boost_LIBRARIES}
                "/home/simon/src/arts/build/src")

if (Boost_FOUND)
  add_executable(run_all run_all.cpp)
  target_link_libraries(run_all ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif (Boost_FOUND)
