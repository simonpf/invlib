cmake_minimum_required (VERSION 2.6)

add_subdirectory(algebra)
add_subdirectory(backends)
add_subdirectory(forward_models)
add_subdirectory(io)
add_subdirectory(mpi)
add_subdirectory(optimization)
add_subdirectory(profiling)
add_subdirectory(sparse)

include_directories("${PROJECT_SOURCE_DIR}/src/"
                    "${PROJECT_SOURCE_DIR}/test/"
                    "${PROJECT_SOURCE_DIR}/ext/")


if (Boost_FOUND)

  include_directories(SYSTEM "${Boost_INCLUDE_DIRS}")

  link_directories(${Boost_LIBRARIES}
                "/home/simon/src/arts/build/src")

  add_executable(run_all run_all.cpp)
#  target_link_libraries(run_all ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

endif (Boost_FOUND)

set(INVLIB_PYTHONPATH ${PROJECT_BINARY_DIR}/src/invlib/interfaces/python)
file(GLOB_RECURSE FILES python "*[!.py]")
foreach(file ${FILES})
  file(RELATIVE_PATH file_rel ${CMAKE_CURRENT_SOURCE_DIR} ${file})
  set(FLOATTYPE float32)
  configure_file("${file_rel}"
                 "${file_rel}_float.py")
  set(FLOATTYPE float64)
  configure_file("${file_rel}"
                 "${file_rel}_double.py")
endforeach(file)
