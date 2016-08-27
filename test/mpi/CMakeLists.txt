cmake_minimum_required (VERSION 2.6)
project (invlib)

include_directories("${PROJECT_SOURCE_DIR}/src/"
                    "${PROJECT_SOURCE_DIR}/test/"
                    ${MPI_INCLUDE_PATH})

# Executables
add_executable(mpi mpi.cpp)
target_link_libraries(mpi ${MPI_LIBRARIES})
add_executable(mpi_reference mpi_reference.cpp)
target_link_libraries(mpi_reference ${MPI_LIBRARIES})
add_executable(mpi_linear linear.cpp)
target_link_libraries(mpi_linear ${MPI_LIBRARIES})
add_executable(mpi_conjugate_gradient conjugate_gradient.cpp)
target_link_libraries(mpi_conjugate_gradient ${MPI_LIBRARIES})
