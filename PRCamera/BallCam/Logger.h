#ifndef LOGGER_H
#define LOGGER_H

#include <fstream>
#include <cstring>

enum MSG_TYPE {
	MSG_DEBUG = 0,
	MSG_INFO,
	MSG_WARN,
	MSG_ERROR
};

class Logger{
public:
	Logger(const char *  name_, MSG_TYPE level=MSG_ERROR);
	~Logger();
	bool Available();
	void Write(const char *  msg, MSG_TYPE mt = MSG_INFO);
	void Error(const char* msg);
	void Warn(const char* msg);
	void Info(const char* msg);
	void Debug(const char* msg);
private:
	std::ofstream log_file;
	char buf_time[32];
	const size_t size_buf = 32;
	MSG_TYPE log_level;
};

#endif