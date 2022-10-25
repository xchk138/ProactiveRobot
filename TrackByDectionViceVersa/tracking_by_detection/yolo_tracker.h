//
// Created by xc on 2022/8/12.
// yolo_tracker.h

#ifndef YOLO_TRACKER_H
#define YOLO_TRACKER_H

#include "config.h"

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "nano_track.h"
#include <cmath>

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
        std::vector<uint64_t>& _uids,
        std::vector<int> & _types,
        std::vector<cv::Rect>& _objs,
        bool _vis = true);

private:
    bool mAvailable;
    bool detectionRequired;

    const char* mPathDetector;
    const char* mPathTrackerBackbone;
    const char* mPathTrackerHead;
    std::vector<int> labels_detect;
    std::vector<int> labels_track;
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
    const cv::Size detect_size = { YOLO_SIZE, YOLO_SIZE };
    const cv::Size track_size = { YOLO_SIZE, YOLO_SIZE };
    // Detection params
    const float cScoreThreshold = 0.37; // 0.37 class score
    const float cNmsThreshold = 0.45;
    const float cConfidenceThreshold = 0.2; // 0.2 object score
    const float cIouThreshold = 0.6;
    const int cOutputCols = num_classes + 5;
    const int cOutputRows = round(YOLO_RATIO*YOLO_SIZE*YOLO_SIZE);
    const char * cOutputNode = YOLO_OUTPUT;

};

/************* tool functions  **************/
void print_id_list(std::vector<int> const& id_list);


#endif // YOLO_TRACKER_H
