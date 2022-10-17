// config.h
#ifndef CONFIG_H
#define CONFIG_H

#include <opencv2/opencv.hpp>

// configuration
#define YOLO_VER 5
#define YOLO_SIZE 224
// different yolo models have various size ratio between input and output
#if YOLO_VER == 5
#define YOLO_RATIO 0.061523438
#define YOLO_OUTPUT "output"
#elif YOLO_VER == 6
#define YOLO_RATIO 0.020507812
#define YOLO_OUTPUT "outputs"
#endif

// global model loading configuration
const static cv::String model_base = "models/";
// YOLO
const static cv::String yolov5_split = "YOLOv5/";
const static cv::String yolov5_main = "yolov5n-224.onnx";
// NanoTrack
const static cv::String nanotrack_split = "NanoTrack/";
const static cv::String nanotrack_backbone = "nanotrack_backbone.onnx";
const static cv::String nanotrack_head = "nanotrack_head.onnx";

// minimum bounding box size and its visualization
const static int min_bbox_area = 15*15;
const static int min_bbox_margin = 10;

// Initialize vectors to hold respective outputs while unwrapping detections.
const static std::vector<std::string> label_names = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                                                "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                                                "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                                                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                                                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                                                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                                                "hair drier", "toothbrush" };

#endif // CONFIG_H