cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(hello)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(hello hello.cpp)
target_link_libraries(hello "${TORCH_LIBRARIES}")
set_property(TARGET hello PROPERTY CXX_STANDARD 14)