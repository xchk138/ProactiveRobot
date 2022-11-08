// utils.h

#ifndef UTILS_H
#define UTILS_H

#include <iostream>

//#define _DEBUG
#define TEST_LOG_TAG "Docking"

#ifdef _WIN32
#define LOGE(fmt, ...) fprintf(stderr, TEST_LOG_TAG "[E] " fmt "\n", __VA_ARGS__); 
#define LOGW(fmt, ...) fprintf(stdout, TEST_LOG_TAG "[W] " fmt "\n", __VA_ARGS__); 
#define LOGI(fmt, ...) fprintf(stdout, TEST_LOG_TAG "[I] " fmt "\n", __VA_ARGS__); 
#else
#define LOGE(fmt, ...) fprintf(stderr, TEST_LOG_TAG "[E] " fmt "\n", ##__VA_ARGS__); 
#define LOGW(fmt, ...) fprintf(stdout, TEST_LOG_TAG "[W] " fmt "\n", ##__VA_ARGS__); 
#define LOGI(fmt, ...) fprintf(stdout, TEST_LOG_TAG "[I] " fmt "\n", ##__VA_ARGS__); 
#endif

#ifdef _DEBUG
#ifdef _WIN32
#define LOGD(fmt, ...) fprintf(stdout, TEST_LOG_TAG "[D] " fmt "\n", __VA_ARGS__); 
#else
#define LOGD(fmt, ...) fprintf(stdout, TEST_LOG_TAG "[D] " fmt "\n", ##__VA_ARGS__); 
#endif
#else
#define LOGD(fmt, ...)
#endif

// cross platform build
#ifndef _WIN32
#define sprintf_s sprintf
#endif

#endif // UTILS_H