cmake_minimum_required( VERSION 3.10 )
set( app_name "tracking_by_detection.a" )
set( src_dir "../../tracking_by_detection" )
project( tracking_by_detection )
set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11" )
include_directories(
	"/usr/local/include/opencv4"
    )
link_directories("opencv")
link_libraries(
    opencv_world
)
add_executable( ${app_name} 
    ${src_dir}/main.cpp 
    ${src_dir}/yolo_tracker.cpp
    ${src_dir}/nano_track.cpp
    )