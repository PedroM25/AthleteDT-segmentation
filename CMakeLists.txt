cmake_minimum_required(VERSION 3.10)
project(PedroAthleteDT)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(PedroAthleteDT main.cpp)

target_link_libraries( PedroAthleteDT ${OpenCV_LIBS} )
