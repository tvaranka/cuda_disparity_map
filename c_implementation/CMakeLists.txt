cmake_minimum_required(VERSION 3.15)

project(stereo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3")
find_package(OpenMP REQUIRED)
add_definitions("-O3")

add_executable(stereo stereo.c ../inc/lodepng.c)
target_link_libraries(stereo m OpenMP::OpenMP_CXX)
