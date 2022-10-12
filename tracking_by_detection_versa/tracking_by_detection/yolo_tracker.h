//
// Created by xc on 2022/8/12.
//

#ifndef YOLO_TRACKER_H
#define YOLO_TRACKER_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "nano_track.h"


uint64_t static GetUniqueObjectId() {
    uint64_t static uid = 0;
    uid++;
    return uid;
}

class YoloTracker {

public:
    struct FaceParams {
        uint64_t faceID;
        cv::Point facePos;
        int faceSize;
    };

public:
    YoloTracker(
        const char* path_detector,
        const char* path_tracker_backbone,
        const char* path_tracker_head);
    ~YoloTracker();
    bool Available();
    bool Track(cv::Mat& _im,
        std::vector<uint64_t>& uids,
        std::vector<cv::Rect>& _objs,
        bool _vis = true);
    bool GetLastResult(std::vector<cv::Rect>& objs);

private:
    bool mAvailable;
    bool detectionRequired;

    const char* mPathDetector;
    const char* mPathTrackerBackbone;
    const char* mPathTrackerHead;
    std::vector<int> class_ids;
    std::vector<cv::Rect> bboxes_detect, bboxes_track;
    std::vector<std::vector<cv::Point2f>> key_pts;
    std::vector<cv::Scalar> colors;
    std::vector<uint64_t> object_ids;
    std::vector<int> offline_time;
    cv::dnn::Net net;
    std::vector<cv::Mat> output_detect;
    int track_time = 0;
    cv::TrackerDaSiamRPN::Params siam_params;
    std::vector<cv::Ptr<cv::TrackerDaSiamRPN>> trackers;
    std::vector<NanoTrack*> nano_trackers;


private:
    void GetRandomColors(
        std::vector<cv::Scalar>& _colors,
        int _size);
    cv::Scalar RandomColor();
    void VisTrackObjs(cv::Mat& _im, cv::Point2d scale);
    void Compute(std::vector<cv::Mat>& outputs, cv::Mat& input_image);
    void PostprocessCoco();
    void PostprocessFace();
    static double BoundingBoxIOU(cv::Rect& bbox1, cv::Rect& bbox2);


    // settings
public:
    const int max_offline_time = 3;
    const int detect_cycle = 30;
    const cv::Size detect_size = { 256, 256 };
    const cv::Size track_size = { 256, 256 };
    // Detection params
    const float SCORE_THRESHOLD = 0.1; // 0.37 class score
    const float NMS_THRESHOLD = 0.45;
    const float CONFIDENCE_THRESHOLD = 0.1; // 0.2 object score
    const float IOU_THE_SAME = 0.6;
    const int YOLO_OUT_ROWS = 4032; // 4032 for 256x256

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

};

/************* tool functions  **************/
void print_id_list(std::vector<int> const& id_list);




#endif // YOLO_TRACKER_H
