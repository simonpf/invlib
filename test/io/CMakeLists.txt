cmake_minimum_required (VERSION 2.6)

include_directories("${PROJECT_SOURCE_DIR}/src/"
                    "${PROJECT_SOURCE_DIR}/test/"
                    "${PROJECT_SOURCE_DIR}/ext/")

enable_testing()

add_executable(read_and_write read_and_write.cpp)
add_test(read_and_write read_and_write)
