#include "StdAfx.h"
#include "Logger.h"
#include <ctime>
#ifdef _WIN32
static void GetLocalTime(char* ts, size_t n) {
    SYSTEMTIME st;
    GetLocalTime(&st); // vc×¨ÓÃ
    sprintf_s(ts, n, "%4d-%02d-%2d %02d:%02d:%02d %03d",
        st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
}
#else
#include <sys/time.h>
static void GetLocalTime(char* time_str, int len) {
    struct tm* ptm;
    char time_string[40];
    long milliseconds;

    timeval tv;
    gettimeofday(&tv, NULL);
    time_t sec_ = tv.tv_sec;
    ptm = localtime(&sec_);

    // Output format: 2018-12-09 10:52:57.200         
    strftime(time_string, sizeof(time_string), "%Y-%m-%d %H:%M:%S", ptm);
    milliseconds = tv.tv_usec / 1000;
    snprintf(time_str, len, "%s.%03ld", time_string, milliseconds);
}
#endif


Logger::Logger(const char * name_, MSG_TYPE level/*=MSG_ERROR*/)
    :log_level(level) {
    buf_time[0] = 0;
	log_file.open(name_, std::ios::ate);
}

Logger::~Logger(){
	log_file.flush();
	log_file.close();
}

bool Logger::Available(){
	return log_file.is_open();
}

void Logger::Write(const char *  msg, MSG_TYPE level/*=MSG_INFO*/){
    if (level >= log_level) {
        GetLocalTime(buf_time, size_buf);
        log_file << buf_time << ": " << msg << std::endl;
    }
}

void Logger::Error(const char* msg) {
    Write(msg, MSG_ERROR);
}

void Logger::Warn(const char* msg) {
    Write(msg, MSG_WARN);
}

void Logger::Info(const char* msg) {
    Write(msg, MSG_INFO);
}

void Logger::Debug(const char* msg) {
    Write(msg, MSG_DEBUG);
}
