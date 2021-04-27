cmake_minimum_required(VERSION 3.0)

project(FACE_NCNN)

add_compile_options(-Wall -fopenmp -O3)
set(CMAKE_CXX_FLAGS "-std=c++11")
include_directories(./src)

set(ncnn_DIR "./lib/cmake/ncnn" CACHE PATH "dir for ncnn.cmake")

#set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.5.1-armlinux/aarch64-linux-gnu/lib/cmake/opencv4)

#set(CMAKE_BUILD_TYPE "DEBUG")
find_package(ncnn REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
aux_source_directory(./src/ all_source)
set(SRC ./face_ncnn.cpp)

add_executable(face_ncnn ${SRC} ${all_source})

target_link_libraries(face_ncnn ncnn ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
