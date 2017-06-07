cmake_minimum_required (VERSION 2.6)

include_directories("${PROJECT_SOURCE_DIR}/src/"
                    "${PROJECT_SOURCE_DIR}/benchmarks/"
                    "${PROJECT_SOURCE_DIR}/ext")

set (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
if  (MKL_FOUND)
  include_directories(${MKL_INCLUDE_DIR})
  add_executable(oem_mkl oem_mkl.cpp)
  target_link_libraries(oem_mkl ${MKL_LIBRARIES})
endif (MKL_FOUND)
