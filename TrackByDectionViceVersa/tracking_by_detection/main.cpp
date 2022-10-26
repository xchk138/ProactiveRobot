#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "yolo_tracker.h"


int main() {
	cv::Mat _im_RAW, _im_crop, _im;
	cv::VideoCapture vid_cap("dock-test.mp4");

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
	cv::Mat im_raw, im_small, im_crop;
	// outputs
	std::vector<uint64_t> uids;
	std::vector<cv::Rect> objs;
	std::vector<int> labels;

	double tick_freq = cv::getTickFrequency();
	double tick_start, tick_stop;

	while(vid_cap.isOpened()){
		vid_cap >> im_raw;
		if (im_raw.empty()) break;
		
		// resize to smaller size for visualization
		int vis_size = 360;
		if (im_raw.cols > im_raw.rows){
			int small_height = vis_size;
			int small_width = vis_size * im_raw.cols / im_raw.rows;
			cv::resize(im_raw, im_small, cv::Size(small_width, small_height));
			int crop_size = (small_width - small_height)/2;
			im_small(cv::Rect(crop_size, 0, vis_size, vis_size)).copyTo(im_crop);
		}else{
			int small_width = vis_size;
			int small_height = vis_size * im_raw.rows / im_raw.cols;
			cv::resize(im_raw, im_small, cv::Size(small_width, small_height));
			int crop_size = (small_height - small_width)/2;
			im_small(cv::Rect(crop_size, 0, vis_size, vis_size)).copyTo(im_crop);
		}

		tick_start = cv::getTickCount();
		// track the objects in current frame
		tracker.Track(im_crop, uids, labels, objs, true);
		tick_stop = cv::getTickCount();
		double cost_ms = (tick_stop - tick_start)*1000 / tick_freq;
		LOGI("FPS: %d", int(1000/cost_ms));
		
		cv::imshow("demo", im_crop);
		int code = cv::waitKey(30);
		if (code == 27 || code==32) break;
	}
	vid_cap.release();
	//system("pause");
	return 0;
}