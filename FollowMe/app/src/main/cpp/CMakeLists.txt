
cmake_minimum_required(VERSION 3.10)

# opencv
#include_directories(${OpenCV_DIR}/jni/include)
#add_library( lib_opencv SHARED IMPORTED )
#set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION ${OpenCV_DIR}/libs/${ANDROID_ABI}/libopencv_java4.so)

set(OpenCV_DIR /home/xc/space1/Downloads/opencv460/OpenCV-android-sdk/sdk/native/jni)
find_package(OpenCV REQUIRED core dnn video)


# ncnn
set(ncnn_DIR /home/xc/space1/Downloads/ncnn-20210525-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(    native-lib
                SHARED
                native-lib.cpp
                yolo_tracker.cpp
                nano_track.cpp
                )

find_library(   log-lib
                log
                )

target_link_libraries(  native-lib
                        ncnn
                        ${OpenCV_LIBS}
                        ${log-lib}
                        )