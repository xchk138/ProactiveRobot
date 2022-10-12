#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "yolo_tracker.h"


int main() {
	cv::Mat _im_RAW, _im_crop, _im;
	cv::VideoCapture vid_cap("docking.mp4");

	// global model loading configuration
	cv::String model_base = "models/";
	// YOLOv5-6.1
	cv::String yolov5_split = "YOLOv5/";
	cv::String yolov5_main = "yolov5n-256.onnx";
	// NanoTrack
	cv::String nanotrack_split = "NanoTrack/";
	cv::String nanotrack_backbone = "nanotrack_backbone.onnx";
	cv::String nanotrack_head = "nanotrack_head.onnx";

	// create a tracker based on YOLO and NanoTrack
	cv::String path_detector = model_base + yolov5_split + yolov5_main;
	cv::String path_tracker_backbone = model_base + nanotrack_split + nanotrack_backbone;
	cv::String path_tracker_head = model_base + nanotrack_split + nanotrack_head;
	YoloTracker tracker(
		path_detector.c_str(),
		path_tracker_backbone.c_str(),
		path_tracker_head.c_str());
	
	if (!tracker.Available()) {
		LOGE("Failed to create Tracker: %s at line %d", __FILE__, __LINE__);
		exit(1);
	}

	// input
	cv::Mat im_RAW;
	// outputs
	std::vector<uint64_t> uids;
	std::vector<cv::Rect> objs;

	double tick_freq = cv::getTickFrequency();
	double tick_start, tick_stop;

	while(vid_cap.isOpened()){
		vid_cap >> im_RAW;
		if (im_RAW.empty()) break;
		
		// resize to smaller size for visualization
		cv::Mat im_small;
		cv::resize(im_RAW, im_small, cv::Size(480, 360));

		tick_start = cv::getTickCount();
		// track the objects in current frame
		tracker.Track(im_small, uids, objs, true);
		tick_stop = cv::getTickCount();
		double cost_ms = (tick_stop - tick_start)*1000 / tick_freq;
		LOGI("FPS: %d", int(1000/cost_ms));
		
		cv::imshow("demo", im_small);
		int code = cv::waitKey(30);
		if (code == 27 || code==32) break;
	}
	vid_cap.release();
	//system("pause");
	return 0;
}