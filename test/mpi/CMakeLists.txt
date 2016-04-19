cmake_minimum_required (VERSION 2.6)
project (invlib)

include_directories("${PROJECT_SOURCE_DIR}/src/"
                    "${PROJECT_SOURCE_DIR}/test/"
                    ${MPI_INCLUDE_PATH})

# Executables
add_executable(mpi mpi.cpp)
target_link_libraries(mpi ${MPI_LIBRARIES})