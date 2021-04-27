# set target machine operating system  eg. Linux  Windows Generic
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")

set(CMAKE_CXX_FLAGS "-std=c++11 -march=native ${CMAKE_CXX_FLAGS}")

#set neon flags
add_definitions(-D__ARM_NEON)

add_definitions(-D__ANDROID__)

SET(ANDROID true)
