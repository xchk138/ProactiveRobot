#pragma once
class nano_track
{
};

#include <vector>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn.hpp>

#define SMALL_SZ  255   //256
#define BIG_SZ  255     //288
#define LR 0.616
#define PENALTY_K 0.007
#define RATIO 1
#define WINDOW_INCLUENCE 0.225
#define PI 3.1415926

using namespace cv;

struct Config {

    std::string windowing = "cosine";
    std::vector<float> window;

    int stride = 16;
    float penalty_k = 0.15;
    float window_influence = 0.476;
    float lr = 0.38;
    int exemplar_size = 127;
    int instance_size = 255;
    int total_stride = 16;
    int score_size = 16;
    float context_amount = 0.5;
    int small_sz = 255;
    int big_sz = 255;
};

struct State {
    int im_h;
    int im_w;
    cv::Scalar channel_ave;
    cv::Point target_pos;
    cv::Point2f target_sz = { 0.f, 0.f };
    float cls_score_max;
};

//static ncnn::Mutex lock;

class NanoTrack {

public:
    NanoTrack();
    ~NanoTrack();

    void init(cv::Mat& img, cv::Rect& bbox);

    void update(const cv::Mat& x_crops,
        cv::Point& target_pos,
        cv::Point2f& target_sz,
        float scale_z,
        float& cls_score_max);

    float track(cv::Mat const& im, cv::Rect& bbox);

    int load_model(
        const char* pathBackbone,
        const char* pathHead);

    bool isInit() { return is_init; }

    bool isAvailable() { return is_available; }

private:
    void create_grids();
    void create_window();
    static cv::Mat get_subwindow_tracking(
        const cv::Mat& im,
        cv::Point2f pos,
        int model_sz,
        int original_sz,
        cv::Scalar channel_ave);

    std::vector<float> grid_to_search_x;
    std::vector<float> grid_to_search_y;
    std::vector<float> window;

    bool is_available = false;
    bool is_init = false;

    cv::dnn::Net net_backbone, net_head;
    cv::Mat zf, xf;

    int stride = 16;
    State state;
    Config cfg;

    // unused
    const float mean_vals[3] = {
            0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f };
    const float norm_vals[3] = {
            1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f };

};


// redefine clearing
void NanoTrackClear(std::vector<NanoTrack*>& trackers);

