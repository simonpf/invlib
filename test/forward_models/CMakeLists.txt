cmake_minimum_required (VERSION 2.6)

include_directories("${PROJECT_SOURCE_DIR}/src/"
                    "${PROJECT_SOURCE_DIR}/test/")

enable_testing()

add_executable(sphere sphere.cpp)
add_test(NAME sphere COMMAND sphere)

add_executable(linear linear.cpp)
add_test(NAME linear COMMAND linear)
