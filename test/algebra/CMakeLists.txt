cmake_minimum_required (VERSION 2.6)
project (maplib)
add_executable(maplib lazy_evaluation.cc)
include_directories ("${PROJECT_SOURCE_DIR}/src/algebra")
