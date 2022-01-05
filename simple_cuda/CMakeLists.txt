cmake_minimum_required(VERSION 3.15)

project(stereo)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

cuda_add_executable(stereo stereo.cu)
target_link_libraries(stereo ${OpenCV_LIBS} ${CUDA_LIBS})
