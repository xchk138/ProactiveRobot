#pragma once
#include <iostream>

//#define _DEBUG

#define TEST_LOG_TAG "Docking"
#define LOGE(fmt, ...) fprintf(stderr, TEST_LOG_TAG "[E] " fmt "\n", __VA_ARGS__); 
#define LOGW(fmt, ...) fprintf(stdout, TEST_LOG_TAG "[W] " fmt "\n", __VA_ARGS__); 
#define LOGI(fmt, ...) fprintf(stdout, TEST_LOG_TAG "[I] " fmt "\n", __VA_ARGS__); 

#ifdef _DEBUG
#define LOGD(fmt, ...) fprintf(stdout, TEST_LOG_TAG "[D] " fmt "\n", __VA_ARGS__); 
#else
#define LOGD(fmt, ...) 
#endif
