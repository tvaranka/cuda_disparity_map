cmake_minimum_required(VERSION 3.15)

project(Stereo)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)
add_definitions("-O3")


add_executable(stereo stereo.c stereo.cl ../inc/lodepng.c)
target_link_libraries(stereo OpenCL::OpenCL OpenMP::OpenMP_CXX)
