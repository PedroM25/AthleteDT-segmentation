cmake_minimum_required(VERSION 3.10)
project(AthleteDT-seg)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(AthleteDT-seg main.cpp)

target_link_libraries( AthleteDT-seg ${OpenCV_LIBS} )
