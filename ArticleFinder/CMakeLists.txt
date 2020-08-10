cmake_minimum_required(VERSION 3.17)
project(ArticleFinder)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package( OpenCV REQUIRED )
set(SOURCE_FILES main.cpp Overlay.cpp Overlay.h Overlay.cpp Overlay.h)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} gflags)



find_package(gflags REQUIRED)

set(CMAKE_CXX_STANDARD 14)
