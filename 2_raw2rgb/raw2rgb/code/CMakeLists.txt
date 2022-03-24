cmake_minimum_required(VERSION 3.16)
project(raw2rgb)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

LINK_DIRECTORIES(/usr/local/lib)

add_executable(raw2rgb main.cpp graw.cpp)
target_link_libraries(raw2rgb ${OpenCV_LIBS})
