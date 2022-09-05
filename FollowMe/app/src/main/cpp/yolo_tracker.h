//
// Created by xc on 2022/8/12.
//

#ifndef HELLOOPENCV_YOLO_TRACKER_H
#define HELLOOPENCV_YOLO_TRACKER_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "nano_track.h"


uint64_t static GetUniqueObjectId(){
    uint64_t static uid = 0;
    uid++;
    return uid;
}

class YoloTracker{

public:
    enum class ROBOT_COMMAND{
        ROBOT_STAY = 0,
        ROBOT_FORWARD,
        ROBOT_BACKWARD,
        ROBOT_LEFT,
        ROBOT_RIGHT
    };

    struct FaceParams{
        uint64_t faceID;
        cv::Point facePos;
        int faceSize;
    };

public:
    YoloTracker(AAssetManager* asset_mgr,
                const char* path_yolo,
                const char* path_siam_main,
                const char* path_siam_cls,
                const char* path_siam_reg);
    ~YoloTracker();
    bool Available() const;
    bool Track(cv::Mat & _im,
               std::vector<uint64_t> & uids,
               std::vector<cv::Rect> & _objs,
               bool _vis = true);
    bool GetLastResult(std::vector<cv::Rect> & objs);
    ROBOT_COMMAND GetRobotCommand(
            uint64_t faceId,
            int faceX,
            int faceY,
            int faceSize);

private:
    bool mAvailable;
    bool detectionRequired;

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

    // nearest face tracking
    FaceParams mFaceParam;
    AAssetManager * mAssetMgr;


private:
    void GetRandomColors(
            std::vector<cv::Scalar> & _colors,
            int _size);
    cv::Scalar RandomColor();
    void VisTrackObjs(cv::Mat & _im, cv::Point2d scale);
    void Compute(std::vector<cv::Mat> & outputs, cv::Mat& input_image);
    void PostprocessCoco();
    void PostprocessFace();
    static double BoundingBoxIOU(cv::Rect & bbox1, cv::Rect & bbox2);



    // settings
public:
    const int max_offline_time = 3;
    const int detect_cycle = 20;
    const cv::Size detect_size = {256, 256};
    const cv::Size track_size = {256,256};
    // Detection params
    const float SCORE_THRESHOLD = 0.37; // class score
    const float NMS_THRESHOLD = 0.45;
    const float CONFIDENCE_THRESHOLD = 0.2; // object score
    const float IOU_THE_SAME = 0.6;
    const int YOLO_OUT_ROWS = 4032;
    // face interaction
    const int MIN_FACE_SIZE = 3000;
    const int MAX_FACE_SIZE = 15000;
    const int MIN_FACE_X = 128-70;
    const int MAX_FACE_X = 128+70;

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


#endif //HELLOOPENCV_YOLO_TRACKER_H
