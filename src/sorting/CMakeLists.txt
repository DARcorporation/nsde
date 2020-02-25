cmake_minimum_required(VERSION 3.14)
project(cpp)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)
add_executable(cpp test.cpp sorting.cpp matrix.cpp)