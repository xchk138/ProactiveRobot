//
// Created by xc on 2022/8/12.
//

#ifndef HELLOOPENCV_UTILS_H
#define HELLOOPENCV_UTILS_H

#include <android/log.h>

#define TEST_LOG_TAG "CamApp"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TEST_LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TEST_LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TEST_LOG_TAG, __VA_ARGS__)

#endif //HELLOOPENCV_UTILS_H
