//
// Created by xc on 2022/8/12.
//
#include "yolo_tracker.h"
#include "utils.h"
#include <opencv2/video/tracking.hpp>
#include <sstream>

void print_id_list(std::vector<int> const& id_list) {
    if (id_list.size() < 2) return;
    char _buf[8];
    std::stringstream ss("");
    ss << "[";
    for (int i = 0; i < int(id_list.size()); ++i) {
        #ifdef _WIN32
        sprintf_s(_buf, " %d", id_list[i]);
        #else
        sprintf(_buf, " %d", id_list[i]);
        #endif
        ss << _buf;
    }
    ss << " ]";
    LOGI("%s", ss.str().c_str());
}


void YoloTracker::GetRandomColors(
    std::vector<cv::Scalar>& _colors,
    int _size
) {
    cv::RNG rng(0);
    for (int i = 0; i < _size; ++i) {
        _colors.emplace_back(rng.uniform(0, 255),
            rng.uniform(0, 255),
            rng.uniform(0, 255));
    }
}


cv::Scalar YoloTracker::RandomColor() {
    cv::RNG rng(cv::getTickCount());
    return { static_cast<double>(rng.uniform(0, 255)),
            static_cast<double>(rng.uniform(0, 255)),
            static_cast<double>(rng.uniform(0, 255)) };
}


void YoloTracker::VisTrackObjs(
    cv::Mat& _im,
    cv::Point2d scale
) {
    if (bboxes_track.size() > colors.size()) {
        LOGE("_bboxes(%d) > _colors(%d)", int(bboxes_track.size()), int(colors.size()));
    }
    for (int i = 0; i < bboxes_track.size(); ++i) {
        cv::Rect _bbox = bboxes_track[i];
        _bbox.x = int(std::floor(scale.x * _bbox.x));
        _bbox.width = int(std::floor(scale.x * _bbox.width));
        _bbox.y = int(std::floor(scale.y * _bbox.y));
        _bbox.height = int(std::floor(scale.y * _bbox.height));
        cv::rectangle(_im, _bbox, colors[i], 3, 8);
    }
}


void YoloTracker::Compute(
    std::vector<cv::Mat>& outputs,
    cv::Mat& input_image)
{
    // Convert to blob.
    cv::Mat blob;
    cv::dnn::blobFromImage(input_image, blob, 1. / 255.,
        input_image.size(), cv::Scalar(), true, false);
    //net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    //net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
    net.setInput(blob);
    net.forward(outputs, "output");
    LOGD("outputs has %d elements, with first %d x %d", int(outputs.size()), int(outputs[0].rows), int(outputs[0].cols));
}


void YoloTracker::PostprocessCoco() {
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<int> classes;
    boxes.clear();
    float* data = (float*)output_detect[0].data;
    const int rows = YOLO_OUT_ROWS;
    // Iterate through all detections.
    class_ids.clear();
    bboxes_detect.clear();

    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float* classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            cv::Mat scores(1, class_names.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire the index of best class  score.
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, nullptr, &max_class_score, nullptr, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD)
            {
                // Store class ID and confidence in the pre-defined respective vectors.
                confidences.push_back(confidence);
                classes.push_back(class_id.x);
                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w));
                int top = int((cy - 0.5 * h));
                int width = int(w);
                int height = int(h);
                // Store good detections in the boxes vector.
                boxes.emplace_back(left, top, width, height);
            }
        }
        // Jump to the next row.
        data += 85;
    }
    // Perform Non-Maximum Suppression and draw predictions.
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

    for (int i = 0; i < indices.size(); i++) {
        std::string _name = class_names[classes[indices[i]]];
        LOGI("%d: %s %.3f%%", i, _name.c_str(), confidences[indices[i]]*100);
        //if (_name == "person") {
            bboxes_detect.push_back(boxes[indices[i]]);
            class_ids.emplace_back(classes[indices[i]]);
        //}
    }
}



// Wideface has a struct like this
// !--- 4 (xywh) --- 1 (obj score) --- 10(5 key points) --- 1 (class score) |
void YoloTracker::PostprocessFace() {
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<cv::Point2f>> pts;

    auto* data = (float*)output_detect[0].data;
    const int rows = YOLO_OUT_ROWS;

    bboxes_detect.clear();
    key_pts.clear();

    // Iterate through all detections.
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4]; // the 5th item for obj score
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float class_score = data[15]; // the last item for class score
            // Continue if the class score is above the threshold.
            if (class_score > SCORE_THRESHOLD)
            {
                // Store class ID and confidence in the pre-defined respective vectors.
                confidences.push_back(confidence);
                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w));
                int top = int((cy - 0.5 * h));
                int width = int(w);
                int height = int(h);
                // Store good detections in the boxes vector.
                boxes.emplace_back(left, top, width, height);
                // get face key points
                pts.emplace_back();
                for (auto _ipt = 0; _ipt < 5; ++_ipt) {
                    pts[pts.size() - 1].emplace_back(cv::Point2f(data[5 + 2 * _ipt], data[5 +
                        2 * _ipt + 1]));
                }
            }
        }
        // Jump to the next row.
        data += 16;
    }
    // Perform Non-Maximum Suppression and draw predictions.
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

    for (int indice : indices) {
        bboxes_detect.emplace_back(boxes[indice]);
        key_pts.emplace_back(pts[indice]);
    }
}


double YoloTracker::BoundingBoxIOU(cv::Rect& bbox1, cv::Rect& bbox2) {
    int x_min = std::max(bbox1.x, bbox2.x);
    int x_max = std::min(bbox1.x + bbox1.width, bbox2.x + bbox2.width);
    int y_min = std::max(bbox1.y, bbox2.y);
    int y_max = std::min(bbox1.y + bbox1.height, bbox2.y + bbox2.height);
    double _base_area = std::min(bbox1.area(), bbox2.area());
    if (x_max <= x_min || y_max <= y_min) return 0;
    else return (y_max - y_min) * (x_max - x_min) / _base_area;
}


// class implementation
YoloTracker::YoloTracker(
    const char* path_detector,
    const char* path_tracker_backbone,
    const char* path_tracker_head){
    // load YOLOv5
    try {
        LOGI("reading Yolo from path: %s", path_detector);
        net = cv::dnn::readNet(path_detector);
        LOGI("reading Yolo done.");
    }
    catch (Exception & ee) {
        LOGE("loading yolov5 error: %s", ee.what());
    }
    if (net.empty()) {
        LOGE("yolov5 NOK!");
        mAvailable = false;
        return;
    }
    LOGI("YOLOv5 model loaded successful");

    // test if nano tracker can be successfully loaded
    NanoTrack _nt;
    int status = _nt.load_model(path_tracker_backbone, path_tracker_head);
    if (status) {
        LOGE("NanoTrack model loaded failed with code: %d!", status);
        return;
    }
    else LOGI("NanoTrack model loaded successful!");

    // save model paths for reuse
    mPathDetector = path_detector;
    mPathTrackerBackbone = path_tracker_backbone;
    mPathTrackerHead = path_tracker_head;

    // enable detection mark
    detectionRequired = true;
    mAvailable = true;
}


bool YoloTracker::Available() {
    return mAvailable;
}


bool YoloTracker::Track(
    cv::Mat& im_raw,
    std::vector<uint64_t>& _uids,
    std::vector<cv::Rect>& _objs,
    bool _vis) {
    if (im_raw.empty()) return false;
    if (!mAvailable) return false;

    // resize image smaller to fit the computation power
    cv::Mat _im_rgb, _im, _im_track;
    cv::resize(im_raw, _im_rgb, detect_size);
    // then delete the alpha channel
    cv::cvtColor(_im_rgb, _im, cv::COLOR_BGRA2BGR);
    cv::resize(_im, _im_track, track_size);
    double scale_track = track_size.width * 1.0 / detect_size.width;

    if (detectionRequired) {
        track_time = 0;
        Compute(output_detect, _im);
        LOGD("YOLO Compute done.");

        bboxes_detect.clear();
        PostprocessCoco();
        LOGD("YOLO Postprocess done.");

        if (bboxes_track.empty()) {
            LOGD("tracking list empty, creating them!");
            NanoTrackClear(nano_trackers);

            for (auto& bb : bboxes_detect)
                bboxes_track.emplace_back(cv::Rect(
                    bb.x * scale_track,
                    bb.y * scale_track,
                    bb.width * scale_track,
                    bb.height * scale_track));
            GetRandomColors(colors, bboxes_track.size());
            for (auto i = 0; i < bboxes_track.size(); ++i) {
                object_ids.push_back(GetUniqueObjectId());
                nano_trackers.emplace_back(new NanoTrack());
                nano_trackers[nano_trackers.size() - 1]->load_model(mPathTrackerBackbone, mPathTrackerHead);
                nano_trackers[nano_trackers.size() - 1]->init(
                    _im_track, bboxes_track[i]);

                offline_time.push_back(0);
            }
            LOGD("tracking list created.");
        }
        else {
            LOGD("tracking list merge detection result...");
            for (auto& bb : bboxes_detect) {
                // convert bb to track scale
                bb.x *= scale_track;
                bb.y *= scale_track;
                bb.width *= scale_track;
                bb.height *= scale_track;

                bool _meet_the_same = false;
                int match_id = -1;

                for (auto j = 0; j < bboxes_track.size(); ++j) {
                    if (BoundingBoxIOU(bb, bboxes_track[j]) >= IOU_THE_SAME) {
                        _meet_the_same = true;
                        match_id = j;
                        break;
                    }
                }
                if (!_meet_the_same) {
                    object_ids.push_back(GetUniqueObjectId());
                    bboxes_track.emplace_back(bb);
                    nano_trackers.emplace_back(new NanoTrack());
                    nano_trackers[nano_trackers.size() - 1]->load_model(mPathTrackerBackbone, mPathTrackerHead);
                    nano_trackers[nano_trackers.size() - 1]->init(
                        _im_track,
                        bboxes_track[bboxes_track.size() - 1]);

                    offline_time.push_back(0);
                    colors.emplace_back(RandomColor());
                }
                else {
                    // replace the tracking one with detection result
                    bboxes_track[match_id] = bb;
                    // and reinitialize the tracker
                    nano_trackers[match_id]->init(
                        _im_track,
                        bboxes_track[match_id]);
                }
            }
            LOGD("tracking list merged.");
        }
        detectionRequired = false;
    }
    else {
        if (!bboxes_track.empty()) {
            LOGD("updating tracking states...");
            std::vector<int> remove_ids;

            for (auto i = 0; i < bboxes_track.size(); ++i) {
                //trackers[i]->update(_im_track, bboxes_track[i]);
                //float _track_score = trackers[i]->getTrackingScore();
                float _track_score = nano_trackers[i]->track(_im_track, bboxes_track[i]);

                LOGD("tracking target #[%d]: %.3f", i, _track_score);
                if (_track_score < 0.95) {
                    offline_time[i] ++;
                }
                else {
                    offline_time[i] = 0;
                }

                for (auto j = i + 1; j < bboxes_track.size(); ++j) {
                    if (BoundingBoxIOU(bboxes_track[i], bboxes_track[j]) >= IOU_THE_SAME) {
                        // reduce duplicates
                        remove_ids.push_back(i);
                        break;
                    }
                }
            }

            // sort remove ids to avoid false removal
            std::sort(remove_ids.begin(), remove_ids.end(), std::greater<int>());
            print_id_list(remove_ids);

            for (int remove_id : remove_ids) {
                object_ids.erase(object_ids.begin() + remove_id);
                bboxes_track.erase(bboxes_track.begin() + remove_id);
                //trackers.erase(trackers.begin() + remove_id);
                delete nano_trackers[remove_id];
                nano_trackers.erase(nano_trackers.begin() + remove_id);
                colors.erase(colors.begin() + remove_id);
                offline_time.erase(offline_time.begin() + remove_id);
            }

            remove_ids.clear();
            for (auto i = 0; i < offline_time.size(); ++i) {
                if (offline_time[i] > max_offline_time) {
                    remove_ids.push_back(i);
                }
            }

            std::sort(remove_ids.begin(), remove_ids.end(), std::greater<int>());
            print_id_list(remove_ids);
            
            for (int remove_id : remove_ids) {
                object_ids.erase(object_ids.begin() + remove_id);
                bboxes_track.erase(bboxes_track.begin() + remove_id);
                //trackers.erase(trackers.begin() + remove_id);
                delete nano_trackers[remove_id];
                nano_trackers.erase(nano_trackers.begin() + remove_id);
                colors.erase(colors.begin() + remove_id);
                offline_time.erase(offline_time.begin() + remove_id);
            }
            LOGD("tracking states updated");
        }
        else detectionRequired = true;
    }

    // update internal timer and check if detection is required
    track_time++;
    if (track_time >= detect_cycle) detectionRequired = true;

    if (_vis) {
        // visualize the tracking objects
        cv::Point2d scale;
        scale.x = im_raw.cols * 1.0 / _im_track.cols;
        scale.y = im_raw.rows * 1.0 / _im_track.rows;
        VisTrackObjs(im_raw, scale);
        //cv::imshow("demo", _im);cv::waitKey(30);
    }

    // copy the results to output list
    _uids.clear();
    _objs.clear();
    for (int _i = 0; _i < bboxes_track.size(); ++_i) {
        _uids.emplace_back(object_ids[_i]);
        _objs.emplace_back(bboxes_track[_i]);
    }

    return true;
}


bool YoloTracker::GetLastResult(std::vector<cv::Rect>& objs) {
    if (!objs.empty()) objs.resize(0);
    std::copy(bboxes_track.begin(), bboxes_track.end(), objs.begin());
    return mAvailable;
}


YoloTracker::~YoloTracker() {
    NanoTrackClear(nano_trackers);
}