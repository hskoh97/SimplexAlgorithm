cmake_minimum_required(VERSION 3.17)
project(Simplex)

set(CMAKE_CXX_STANDARD 14)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(Simplex main.cpp linear_program.cpp linear_program.h)
target_link_libraries(Simplex ${CONAN_LIBS})