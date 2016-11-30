cmake_minimum_required (VERSION 2.6)
project (invlib)

if (EIGEN3_INCLUDE_DIR)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
endif (EIGEN3_INCLUDE_DIR)

include_directories("${PROJECT_SOURCE_DIR}/src/"
                    "${PROJECT_SOURCE_DIR}/test/"
                    "${MPI_INCLUDE_PATH}")

# Executables
if (EIGEN3_FOUND)
    add_executable(oem oem.cpp)
    add_executable(oem_preconditioned oem_preconditioned.cpp)
    add_executable(oem_mpi oem_mpi.cpp)
    target_link_libraries(oem_mpi ${MPI_LIBRARIES})
    add_executable(oem_preconditioned_mpi oem_preconditioned_mpi.cpp)
    target_link_libraries(oem_preconditioned_mpi ${MPI_LIBRARIES})
endif (EIGEN3_FOUND)
