cmake_minimum_required (VERSION 2.6)

if (MKL_MINIMAL_LIBRARY)
  # Executables
  include_directories(
    ${MPI_INCLUDE_PATH}
    "${PROJECT_SOURCE_DIR}/src/ "
    ${MKL_INCLUDE_DIR})

  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  set (archs Cpu)
  if (MPI_CXX_FOUND)
    list(APPEND archs Mpi)
  endif (MPI_CXX_FOUND)

  foreach(p float double)
    foreach (a ${archs})

      set(libraries ${MKL_MINIMAL_LIBRARY})
      if (${a} STREQUAL Mpi)
        list(APPEND libraries ${MPI_LIBRARIES})
      endif(${a} STREQUAL Mpi)

      set(PREC ${p})
      set(ARCH ${a})
      string(TOLOWER invlib_${a}_${p} libname)
      configure_file(invlib.cpp ${libname}.cpp)
      add_library (${libname} SHARED ${libname})
      target_link_libraries(${libname} ${libraries})
    endforeach(a)
  endforeach(p)

  #
  # Copy python source files.
  #

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

  set(LIBINVLIB_PATH ${CMAKE_CURRENT_BINARY_DIR})

  foreach(p float32 float64)
    foreach(a cpu mpi)
      set(PREC ${p})
      set(ARCH ${a})
      set(filename ${a}_${p})
      if (PYTHON_OUTPUT_DIR)
        configure_file(invlib/api/template "${PYTHON_OUTPUT_DIR}/invlib/api/${filename}.py")
      else (PYTHON_OUTPUT_DIR)
        configure_file(invlib/api/template invlib/api/${filename}.py)
      endif (PYTHON_OUTPUT_DIR)
    endforeach(a)
  endforeach(p)
endif (MKL_MINIMAL_LIBRARY)
