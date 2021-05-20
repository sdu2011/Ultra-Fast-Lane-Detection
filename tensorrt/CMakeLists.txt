cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(lane_detection)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

find_package(OpenCV REQUIRED)

file(GLOB SOURCE_FILES
"src/*.cpp")
add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES})


target_link_libraries(${CMAKE_PROJECT_NAME} 
${OpenCV_LIBS})


