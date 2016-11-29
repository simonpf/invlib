cmake_minimum_required (VERSION 2.6)

include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
include_directories("${PROJECT_SOURCE_DIR}/src/"
                    "${PROJECT_SOURCE_DIR}/test/"
                    "${PROJECT_SOURCE_DIR}/ext/pugixml"
                    /opt/intel/mkl/include)

# Executables
if(MATS_DATA)
  add_definitions(-DMATS_DATA=${MATS_DATA})
  add_executable(mats mats.cpp)
  target_link_libraries(mats
                        /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a
                        /opt/intel/mkl/lib/intel64/libmkl_core.a
                        /opt/intel/mkl/lib/intel64/libmkl_intel_thread.a
                        /opt/intel/mkl/lib/intel64/libmkl_core.a
                        /opt/intel/lib/intel64/libiomp5.a
                        pthread
                        dl)
endif(MATS_DATA)

cuda_add_executable(matrix_vector_multiplication matrix_vector_multiplication.cpp)
target_link_libraries(matrix_vector_multiplication
                      ${CUDA_CUBLAS_LIBRARIES}
                      ${CUDA_cusparse_LIBRARY}
                      /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a
                      /opt/intel/mkl/lib/intel64/libmkl_core.a
                      /opt/intel/mkl/lib/intel64/libmkl_intel_thread.a
                      /opt/intel/mkl/lib/intel64/libmkl_core.a
                      /opt/intel/lib/intel64/libiomp5.a
                      pthread)