#include <iostream>
#include <opencv.hpp>
#include "utils.h"
#include "yolo_tracker.h"

// Detection params
const float INPUT_WIDTH = 480.0;
const float INPUT_HEIGHT = 480.0;
const float SCORE_THRESHOLD = 0.1;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.1;
const float IOU_THE_SAME = 0.6;


void GetRandomColors(
	std::vector<cv::Scalar> & _colors,
	int _size
) {
	cv::RNG rng(0);
	for (int i = 0; i < _size; ++i) {
		_colors.push_back(cv::Scalar(rng.uniform(0, 255),
			rng.uniform(0, 255), rng.uniform(0, 255)));
	}
}

cv::Scalar RandomColor() {
	cv::RNG rng(cv::getTickCount());
	return cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
}

void VisTrackObjs(
	cv::Mat & _im,
	std::vector<cv::Rect> const & _bboxes, 
	std::vector<cv::Scalar> const & _colors,
	std::vector<int> const & offline_time
) {
	if (_bboxes.size() < _colors.size()) {
		LOGE("_bboxes(%lu) > _colors(%lu)", _bboxes.size(), _colors.size());
	}
	for (int i = 0; i < _bboxes.size(); ++i) {
		//if(offline_time[i] <= 0) // 离线目标不显示
			cv::rectangle(_im, _bboxes[i], _colors[i], 2, 8);
	}
}

// preprocess the DNN input
void Compute(std::vector<cv::Mat> & outputs, 
	cv::Mat& input_image, 
	cv::dnn::Net& net)
{
	// Convert to blob.
	cv::Mat blob;
	cv::dnn::blobFromImage(input_image, blob, 1. / 255., 
		cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
	net.setInput(blob);
	net.forward(outputs, net.getUnconnectedOutLayersNames());
}


void Postprocess(
	std::vector<cv::Rect> & output_bboxes, 
	std::vector<cv::Mat> & net_outputs, 
	cv::Size raw_size,
	int num_classes
){
	// Initialize vectors to hold respective outputs while unwrapping detections.
	const std::vector<std::string> class_names = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
		"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
		"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
		"hair drier", "toothbrush" };
	std::vector<int> class_ids;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	boxes.clear();
	// Resizing factor.
	float x_factor = raw_size.width / INPUT_WIDTH;
	float y_factor = raw_size.height / INPUT_HEIGHT;
	float* data = (float*)net_outputs[0].data;
	const int dimensions = 85;
	// 25200 for default size 480.
	const int rows = 14175;
	// Iterate through all detections.
	for (int i = 0; i < rows; ++i)
	{
		float confidence = data[4];
		// Discard bad detections and continue.
		if (confidence >= CONFIDENCE_THRESHOLD)
		{
			float* classes_scores = data + 5;
			// Create a 1x85 Mat and store class scores of 80 classes.
			cv::Mat scores(1, num_classes, CV_32FC1, classes_scores);
			// Perform minMaxLoc and acquire the index of best class  score.
			cv::Point class_id;
			double max_class_score;
			cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
			// Continue if the class score is above the threshold.
			if (max_class_score > SCORE_THRESHOLD)
			{
				// Store class ID and confidence in the pre-defined respective vectors.
				confidences.push_back(confidence);
				class_ids.push_back(class_id.x);
				// Center.
				float cx = data[0];
				float cy = data[1];
				// Box dimension.
				float w = data[2];
				float h = data[3];
				// Bounding box coordinates.
				int left = int((cx - 0.5 * w) * x_factor);
				int top = int((cy - 0.5 * h) * y_factor);
				int width = int(w * x_factor);
				int height = int(h * y_factor);
				// Store good detections in the boxes vector.
				boxes.push_back(cv::Rect(left, top, width, height));
			}
		}
		// Jump to the next row.
		data += 85;
	}
	// Perform Non-Maximum Suppression and draw predictions.
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
	output_bboxes.clear();
	for (int i = 0; i < indices.size(); i++) {
		std::string _name = class_names[class_ids[indices[i]]];
		LOGI("%d: %s", i, _name.c_str());
		if(_name == "person") 
			output_bboxes.push_back(boxes[indices[i]]);
	}
}


float BoundingBoxIOU(cv::Rect & bbox1, cv::Rect & bbox2) {
	int x_min = std::max(bbox1.x, bbox2.x);
	int x_max = std::min(bbox1.x + bbox1.width, bbox2.x + bbox2.width);
	int y_min = std::max(bbox1.y, bbox2.y);
	int y_max = std::min(bbox1.y + bbox1.height, bbox2.y + bbox2.height);
	float _base_area = std::min(bbox1.area(), bbox2.area());
	if (x_max <= x_min) {
		return 0;
	}
	else if (y_max <= y_min) {
		return 0;
	}
	else return (y_max - y_min) * (x_max - x_min) / _base_area;
}


int64_t static GetUniqueObjectId(){
	int64_t static uid = 0;
	uid++;
	return uid;
}


int main() {
	cv::Mat _im_RAW, _im_crop, _im;
	cv::VideoCapture vid_cap("docking.mp4");
	bool detectionRequired = true;
	std::vector<cv::Rect> bboxes_detect, bboxes_track;
	std::vector<cv::Scalar> colors;
	std::vector<uint64_t> object_ids;
	// 各个目标的离线时长，当超过阈值，剔除，离线时长大于0的都不会显示在图中
	std::vector<int> offline_time; 
	// 10秒都没恢复就剔除该潜在目标
	const int max_offline_time = 3; 

	// global model loading configuration
	cv::String model_base = "models/";
	// DaSiamRPN
	cv::String siam_split = "SiamRPN/";
	cv::String siam_main = "dasiamrpn_model.onnx";
	cv::String siam_cls = "dasiamrpn_kernel_cls1.onnx";
	cv::String siam_reg = "dasiamrpn_kernel_r1.onnx";
	// YOLOv5-6.1
	cv::String yolov5_split = "YOLOv5/";
	cv::String yolov5_main = "yolov5n.onnx";

	// load YOLOv5
	cv::dnn::Net net;
	try {
		net = cv::dnn::readNet(model_base + yolov5_split + yolov5_main);
	}
	catch (const cv::Exception& ee) {
		std::cerr << "Exception: " << ee.what() << std::endl;
		LOGE("Exception in loading YOLOv5 model: %s", ee.what());
		return 1;
	}
	
	std::vector<cv::Mat> output_detect;
	if (net.empty()) LOGE("not okay!");
	LOGI("model loaded okay");
	
	// using DNN model to track targets
	cv::TrackerDaSiamRPN::Params siam_params;
	std::vector<cv::Ptr<cv::TrackerDaSiamRPN>> trackers;
	try {
		siam_params.model = cv::samples::findFile(model_base + siam_split + siam_main);
		siam_params.kernel_cls1 = cv::samples::findFile(model_base + siam_split + siam_cls);
		siam_params.kernel_r1 = cv::samples::findFile(model_base + siam_split + siam_reg);
		siam_params.backend = 0;
		siam_params.target = 0;
		// test if a DaSiamRPN tracker can be created
		//basic_tracker = cv::TrackerDaSiamRPN::create(params);

		const char* nanotrack_backbone = "models/NanoTrack/nanotrack_backbone.onnx";
		const char* nanotrack_head = "models/NanoTrack/nanotrack_head.onnx";

	}
	catch (const cv::Exception& ee){
		std::cerr << "Exception: " << ee.what() << std::endl;
		LOGE("Exception in loading SiamRPN model: %s", ee.what());
		return 1;
	}

	//cv::Ptr<cv::TrackerMIL> tracker = cv::TrackerMIL::create();
	int track_time = 0;
	const int detect_cycle = 20;
	while(vid_cap.isOpened()){
		vid_cap >> _im_RAW;
		if (_im_RAW.empty()) break;
		int _strip = INPUT_HEIGHT * 1.0 * _im_RAW.cols / _im_RAW.rows;
		cv::resize(_im_RAW, _im_crop, cv::Size(_strip, INPUT_HEIGHT));
		int _margin = (_strip - INPUT_WIDTH) / 2;
		_im = _im_crop(cv::Rect(_margin, 0, INPUT_WIDTH, INPUT_HEIGHT)).clone();
		if(detectionRequired){
			track_time = 0;
			// 获取跟踪的目标检测框
			//cv::selectROIs("矩形框选择要跟踪的对象", _im, bboxes_detect, true, false);
			Compute(output_detect, _im, net);
			LOGI("model compute okay");
			std::cout << output_detect[0].size() << std::endl;
			LOGI("YOLOv5 output: %d %d x %d", output_detect[0].empty(), output_detect[0].rows, output_detect[0].cols);
			
			bboxes_detect.clear();
			Postprocess(bboxes_detect, output_detect, _im.size(), 80);
			std::cout << bboxes_detect.size() << std::endl;

			if (bboxes_track.empty()) {
				trackers.clear();
				// 这是初始帧，检测是必须的
				bboxes_track = bboxes_detect;
				// 此时需要重新生成目标颜色区分
				GetRandomColors(colors, bboxes_track.size());
				// 并用这些检测框分别初始化各个跟踪器，每个目标都分配一个跟踪器
				for (auto i = 0; i < bboxes_track.size(); ++i) {
					object_ids.push_back(GetUniqueObjectId());
					trackers.push_back(cv::TrackerDaSiamRPN::create(siam_params));
					trackers[i]->init(_im, bboxes_track[i]);
					offline_time.push_back(0); // 各个目标都处于在线状态
				}
			}
			else {
				// 做闭环控制，并且引入可能新出现的目标
				// case 1: 检测结果和在线目标重合
				// case 2: 检测结果不在跟踪列表中，此时新增跟踪目标
				// case 3: 检测中找不到匹配跟踪的目标，则继续跟踪该目标
				for (auto i = 0; i < bboxes_detect.size(); ++i) {
					// 计算该检测目标框和每个跟踪目标框的IOU，如果达到阈值，则认为是同一目标
					bool _meet_the_same = false;
					int match_id = -1;
					for(auto j=0; j<bboxes_track.size(); ++j){
						if (BoundingBoxIOU(bboxes_detect[i], bboxes_track[j]) >= IOU_THE_SAME) {
							// 交叉区域足够大，认为是同一个目标
							_meet_the_same = true;
							match_id = j;
							break;
						}
					}
					if (!_meet_the_same) {
						// 检测中出现了新目标
						object_ids.push_back(GetUniqueObjectId());
						bboxes_track.push_back(bboxes_detect[i]);
						trackers.push_back(cv::TrackerDaSiamRPN::create(siam_params));
						trackers[trackers.size()-1]->init(_im, bboxes_track[bboxes_track.size()-1]);
						offline_time.push_back(0); // 各个目标都处于在线状态
						colors.push_back(RandomColor());
					}
					else {
						//// 如果找到了匹配的跟踪目标，但是它在离线列表中，则恢复该旧目标
						//if (offline_time[match_id] > 0) {
						//	offline_time[match_id] = 0;
						//}
					}
				}
			}
			detectionRequired = false;
		}
		else{
			if (bboxes_track.size() > 0) {
				// 跟踪每一个目标，并更新其矩形框
				for(auto i=0; i<bboxes_track.size();++i){
					//if (offline_time[i] > 0) continue;
					trackers[i]->update(_im, bboxes_track[i]);
					float _track_score = trackers[i]->getTrackingScore();
					LOGI("tracking target #[%d]: %.3f", i, _track_score);
					if (_track_score < 0.95) {
						//detectionRequired = true;
						// 设定离线时间自增
						offline_time[i] ++;
					}
					else
					{
						// 重置在线状态
						offline_time[i] = 0;
					}
				}
				// 对于跟踪目标列表中的每个目标检查离线时间
				// 如果离线超过时长限制，则剔除该目标
				std::vector<int> remove_ids;
				remove_ids.clear();
				for (auto i = 0; i < offline_time.size(); ++i) {
					if (offline_time[i] > max_offline_time) {
						// 长时间没看到该目标出现，则去除它
						remove_ids.push_back(i);
					}
				}
				// 根据待删除列表进行删除
				std::sort(remove_ids.begin(), remove_ids.end(), std::greater<>());
				print_id_list(remove_ids);

				for(auto i=0; i<remove_ids.size(); ++i){
					object_ids.erase(object_ids.begin() + remove_ids[i]);
					bboxes_track.erase(bboxes_track.begin() + remove_ids[i]);
					trackers.erase(trackers.begin() + remove_ids[i]);
					colors.erase(colors.begin() + remove_ids[i]);
					offline_time.erase(offline_time.begin() + remove_ids[i]);
				}
			}
			else detectionRequired = true;
		}
		// 可视化所有目标
		VisTrackObjs(_im, bboxes_track, colors, offline_time);
		track_time++;
		if (track_time >= detect_cycle) detectionRequired = true;

		cv::imshow("demo", _im);
		int code = cv::waitKey(30);
		if (code == 27 || code==32) break;
	}
	vid_cap.release();
	system("pause");
	return 0;
}