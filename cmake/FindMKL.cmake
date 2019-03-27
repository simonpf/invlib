# Find the MKL libraries
#
# Options:
#
#   MKL_STATIC        :   use static linking
#   MKL_SDL           :   Single Dynamic Library interface
#
# This module defines the following variables:
#
#   MKL_FOUND            : True if MKL_INCLUDE_DIR are found
#   MKL_INCLUDE_DIR      : where to find mkl.h, etc.
#   MKL_INCLUDE_DIRS     : set when MKL_INCLUDE_DIR found
#   MKL_LIBRARIES        : the library to link against.

include(FindPackageHandleStandardArgs)

set(MKLROOT "$ENV{MKLROOT}" CACHE PATH "Folder containing MKL.")

if (NOT "$ENV{INTELROOT}" STREQUAL "")
  set(INTELROOT "$ENV{INTELROOT}" CACHE INTERNAL "Folder containing intel libs.")
else()
  find_path(INTELROOT bin/compilervars.sh PATHS ${MKLROOT}/..)
endif()

message(intelroot: ${INTELROOT})

set(INTELROOT "$ENV{INTELROOT}" CACHE PATH "Folder contains intel libs")

# Find include dir
find_path(MKL_INCLUDE_DIR mkl.h PATHS ${MKLROOT}/include NO_DEFAULT_PATH)
if(NOT MKL_INCLUDE_DIR)
  find_path(MKL_INCLUDE_DIR mkl.h PATHS ENV CPATH)
endif(NOT MKL_INCLUDE_DIR)

# Handle suffix
set(_MKL_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})

if(MKL_STATIC)
  set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
else()
  set(CMAKE_FIND_LIBRARY_SUFFIXES .so)
endif()

set (MKL_ARCHITECTURE 64)
if      (${MKL_ARCHITECTURE} EQUAL 32)
  set(MKL_LIBRARY_FOLDER intel32_lin)
elseif  (${MKL_ARCHITECTURE} EQUAL 64)
  set(MKL_LIBRARY_FOLDER intel64_lin)
  set(MKL_LIBRARY _lp64)
elseif  (${MKL_ARCHITECTURE} EQUAL MIC)
  set(MKL_LIBRARY_FOLDER intel64_lin_mic)
  set(MKL_LIBRARY _lp64)
endif   (${MKL_ARCHITECTURE} EQUAL 32)

if(MKL_SDL)
  find_library(MKL_LIBRARY mkl_rt PATHS ${MKLROOT}/lib/${MKL_LIBRARY_FOLDER}/
                                  ENV   LIBRARY_PATH )
    set(MKL_MINIMAL_LIBRARY ${MKL_LIBRARY})
else()
    ######################### Interface layer #######################
    set(MKL_INTERFACE_LIBRARY_NAME mkl_intel${MKL_LIBRARY})
    find_library(MKL_INTERFACE_LIBRARY ${MKL_INTERFACE_LIBRARY_NAME}
      PATHS ${MKLROOT}/lib/${MKL_LIBRARY_FOLDER}/ ${MKL_INTERFACE_LIBRARY}
      ENV LIBRARY_PATH)

    ######################## Threading layer ########################
    set(MKL_THREADING_LIBRARY_NAME mkl_intel_thread)
    find_library(MKL_THREADING_LIBRARY ${MKL_THREADING_LIBRARY_NAME}
      PATHS ${MKLROOT}/lib/${MKL_LIBRARY_FOLDER}/
      ENV LIBRARY_PATH)

    ####################### Computational layer #####################
    find_library(MKL_CORE_LIBRARY mkl_core
      PATHS ${MKLROOT}/lib/${MKL_LIBRARY_FOLDER}/
      ENV LIBRARY_PATH)
    find_library(MKL_FFT_LIBRARY mkl_cdft_core
      PATHS ${MKLROOT}/lib/${MKL_LIBRARY_FOLDER}/
      ENV LIBRARY_PATH)
    find_library(MKL_SCALAPACK_LIBRARY mkl_scalapack${MKL_LIBRARY}
      PATHS ${MKLROOT}/lib/${MKL_LIBRARY_FOLDER}/
      ENV LIBRARY_PATH)
    find_library(MKL_BLACS_LIBRARY mkl_blacs${MKL_LIBRARY}
      PATHS ${MKLROOT}/lib/${MKL_LIBRARY_FOLDER}/
      ENV LIBRARY_PATH)
    find_library(MKL_AVX_LIBRARY mkl_avx
      PATHS ${MKLROOT}/lib/${MKL_LIBRARY_FOLDER}/
      ENV LIBRARY_PATH)

    find_library(MKL_AVX2_LIBRARY mkl_avx2
      PATHS ${MKLROOT}/lib/${MKL_LIBRARY_FOLDER}/
      ENV LIBRARY_PATH)

    find_library(MKL_AVX512_LIBRARY mkl_avx512
      PATHS ${MKLROOT}/lib/${MKL_LIBRARY_FOLDER}/
      ENV LIBRARY_PATH)

    find_library(MKL_DEF_LIBRARY mkl_def
      PATHS ${MKLROOT}/lib/${MKL_LIBRARY_FOLDER}/
      ENV LIBRARY_PATH)

    ############################ RTL layer ##########################
    find_library(MKL_RTL_LIBRARY iomp5
      PATHS ${INTELROOT}/lib/${MKL_LIBRARY_FOLDER}/
      ENV LIBRARY_PATH)

    set(MKL_LIBRARY ${MKL_INTERFACE_LIBRARY} ${MKL_THREADING_LIBRARY}
                    ${MKL_CORE_LIBRARY} ${MKL_FFT_LIBRARY}
                    ${MKL_SCALAPACK_LIBRARY}  ${MKL_RTL_LIBRARY} ${MKL_AVX_LIBRARY})
    set(MKL_MINIMAL_LIBRARY ${MKL_INTERFACE_LIBRARY} ${MKL_THREADING_LIBRARY}
      ${MKL_CORE_LIBRARY} ${MKL_RTL_LIBRARY} ${MKL_AVX_LIBRARY} ${MKL_AVX2_LIBRARY}
      ${MKL_AVX_512} ${MKL_DEF_LIBRARY})
endif()

set(CMAKE_FIND_LIBRARY_SUFFIXES ${_MKL_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})

find_package_handle_standard_args(MKL DEFAULT_MSG
    MKL_INCLUDE_DIR MKL_LIBRARY MKL_MINIMAL_LIBRARY)

if(MKL_FOUND)
    set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
    set(MKL_LIBRARIES ${MKL_LIBRARY})
    set(MKL_MINIMAL_LIBRARIES ${MKL_MINIMAL_LIBRARY})
endif()

if ((NOT MKL_SCALAPACK_LIBRARY) OR (NOT MKL_BLACS_LIBRARY))
  set(MKL_FOUND FALSE)
endif()
