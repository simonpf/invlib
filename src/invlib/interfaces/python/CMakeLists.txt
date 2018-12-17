cmake_minimum_required (VERSION 2.6)

# Executables
find_package(Python COMPONENTS Development)
find_package(Numpy)

if (Python_FOUND)
  include_directories(
    ${Python_INCLUDE_DIR}
    "${PROJECT_SOURCE_DIR}/src/ "
    ${numpy_INCLUDE_DIR})

  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  add_library (invlib SHARED invlib.cpp)
  include_directories(${MKL_INCLUDE_DIR})
  MESSAGE(WARNING ${MKL_MINIMAL_LIBRARY})
  target_link_libraries(invlib ${BLAS_LIBRARIES} ${MKL_MINIMAL_LIBRARY})

  #
  # Copy python source files.
  #

  set(LIBINVLIB_PATH ${CMAKE_CURRENT_BINARY_DIR})
  file(GLOB_RECURSE FILES invlib "*.py")
  foreach(file ${FILES})
    file(RELATIVE_PATH file_rel ${CMAKE_CURRENT_SOURCE_DIR} ${file})
    configure_file("${file_rel}"
                   "${file_rel}")
  endforeach(file)

endif (Python_FOUND)
