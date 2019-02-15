cmake_minimum_required (VERSION 2.6)

if (MKL_MINIMAL_LIBRARY)
  # Executables
  include_directories(
    ${Python_INCLUDE_DIR}
    "${PROJECT_SOURCE_DIR}/src/ "
    ${MKL_INCLUDE_DIR})

  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  set(FLOATTYPE float)
  configure_file(invlib.cpp invlib_float.cpp)
  add_library (invlib_float SHARED invlib_float.cpp)
  target_link_libraries(invlib_float ${BLAS_LIBRARIES} ${MKL_MINIMAL_LIBRARY})

  set(FLOATTYPE double)
  configure_file(invlib.cpp invlib_double.cpp)
  add_library (invlib_double SHARED invlib_double.cpp)
  target_link_libraries(invlib_double ${BLAS_LIBRARIES} ${MKL_MINIMAL_LIBRARY})

  #
  # Copy python source files.
  #

  set(LIBINVLIB_PATH ${CMAKE_CURRENT_BINARY_DIR})
  file(GLOB_RECURSE FILES invlib "*.py")

  foreach(file ${FILES})
    file(RELATIVE_PATH file_rel ${CMAKE_CURRENT_SOURCE_DIR} ${file})

    if (PYTHON_OUTPUT_DIR)
      configure_file("${file_rel}"
                  "${PYTHON_OUTPUT_DIR}/${file_rel}")
      message("${PYTHON_OUTPUT_DIR}/${file_rel}")
    else (PYTHON_OUTPUT_DIR)
      configure_file("${file_rel}"
                    "${file_rel}")
    endif (PYTHON_OUTPUT_DIR)
  endforeach(file)

  set(FLOATTYPE float32)
  if (PYTHON_OUTPUT_DIR)
    configure_file(invlib/api/template "${PYTHON_OUTPUT_DIR}/invlib/api/float.py")
  else (PYTHON_OUTPUT_DIR)
    configure_file(invlib/api/template invlib/api/float.py)
  endif (PYTHON_OUTPUT_DIR)

  set(FLOATTYPE float64)
  if (PYTHON_OUTPUT_DIR)
    configure_file(invlib/api/template "${PYTHON_OUTPUT_DIR}/invlib/api/double.py")
  else (PYTHON_OUTPUT_DIR)
    configure_file(invlib/api/template invlib/api/double.py)
  endif (PYTHON_OUTPUT_DIR)
endif(MKL_MINIMAL_LIBRARY)
