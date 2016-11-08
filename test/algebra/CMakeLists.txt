cmake_minimum_required (VERSION 2.6)
project (invlib)

include_directories(SYSTEM "${Boost_INCLUDE_DIRS}")
include_directories("${PROJECT_SOURCE_DIR}/src/"
                    "${PROJECT_SOURCE_DIR}/test/")

link_directories(${Boost_LIBRARIES})

# Executables
if (Boost_FOUND)
  add_executable(identities identities.cpp)
  target_link_libraries(identities ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
  add_executable(solvers solvers.cpp)
  target_link_libraries(solvers ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
  add_executable(precision_matrix precision_matrix.cpp)
  target_link_libraries(precision_matrix ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
  add_executable(transformation transformation.cpp)
  target_link_libraries(transformation ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
  add_executable(diagonal diagonal.cpp)
  target_link_libraries(transformation ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif (Boost_FOUND)