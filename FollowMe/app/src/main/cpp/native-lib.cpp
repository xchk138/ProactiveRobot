#include <jni.h>
#include "opencv2/core/core.hpp"
#include <android/asset_manager_jni.h>

#include "utils.h"
#include "yolo_tracker.h"


extern "C" {

jlong JNICALL Java_ph_edu_dlsu_robot_MainActivity_jniGetYoloTracker(
        JNIEnv *env,
        jobject instance,
        jobject assets,
        jstring pathYolo,
        jstring pathSiamMain,
        jstring pathSiamCls,
        jstring pathSiamReg) {
    // convert from java memory into C memory
    AAssetManager * assetMgr = AAssetManager_fromJava(env, assets);
    int sYolo = env->GetStringLength(pathYolo);
    char * pYolo = new char[sYolo];
    env->GetStringUTFRegion(pathYolo, 0, sYolo, pYolo);

    int sSiamMain = env->GetStringLength(pathSiamMain);
    char * pSiamMain = new char[sSiamMain];
    env->GetStringUTFRegion(pathSiamMain, 0, sSiamMain, pSiamMain);

    int sSiamCls = env->GetStringLength(pathSiamCls);
    char * pSiamCls = new char[sSiamCls];
    env->GetStringUTFRegion(pathSiamCls, 0, sSiamCls, pSiamCls);

    int sSiamReg = env->GetStringLength(pathSiamReg);
    char * pSiamReg = new char[sSiamReg];
    env->GetStringUTFRegion(pathSiamReg, 0, sSiamReg, pSiamReg);

    YoloTracker * tracker = new YoloTracker(
            assetMgr, pYolo, pSiamMain, pSiamCls, pSiamReg);

    // release
    delete [] pYolo;
    delete [] pSiamMain;
    delete [] pSiamCls;
    delete [] pSiamReg;

    return reinterpret_cast<jlong>(tracker);
}
};

extern "C" {

void JNICALL
Java_ph_edu_dlsu_robot_MainActivity_jniFreeYoloTracker(
        JNIEnv *env,
        jobject instance,
        jlong tracker_addr) {
    if (tracker_addr!=0)
        delete (YoloTracker *) tracker_addr;
}
};

extern "C" {

jint JNICALL
Java_ph_edu_dlsu_robot_MainActivity_jniTrackAll(
        JNIEnv *env,
        jobject instance,
        jlong pFrame,
        jlong pTracker) {
    double tick_freq = cv::getTickFrequency();
    double tick_start = cv::getTickCount();

    // in case front camera, do flip the view
    cv::Mat &img = *(cv::Mat *) pFrame;
    cv::flip(img, img, -1);
    cv::rotate(img, img, cv::ROTATE_90_CLOCKWISE);

    //LOGI("image size: w=%d, h=%d", img.cols, img.rows);

    // track with visualization only
    YoloTracker & tracker = *(YoloTracker*)(pTracker);
    std::vector<cv::Rect> objs;
    std::vector<uint64_t> uids;

    if(tracker.Available()) tracker.Track(img, uids, objs);
    cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);

    double tick_stop = cv::getTickCount();
    double cost_time = 1000.0 * (tick_stop - tick_start) / tick_freq;
    LOGI("FPS: %d ", int(1000.0 / cost_time));

    if (objs.empty()) return static_cast<jint>(YoloTracker::ROBOT_COMMAND::ROBOT_STAY);

    // select the nearest face as the interactive target
    // attention: the image is 90-degree orientation
    int area_max = 0;
    int local_id = 0;
    for(auto i=0; i<objs.size(); ++i){
        int _area = objs[i].area();
        if(_area > area_max) {
            area_max = _area;
            local_id = i;
        }
    }

    // log the face info
    LOGI("face ID: %lu; face size: %d; face pos: (%d, %d)", \
        uids[local_id], \
        area_max, \
        objs[local_id].x + int(0.5*objs[local_id].width), \
         objs[local_id].y + int(0.5*objs[local_id].height));

    // get the corresponding command with current face info, and
    // update biggest face id and its center X,Y and area size
    YoloTracker::ROBOT_COMMAND cmd = tracker.GetRobotCommand(
            uids[local_id],
            objs[local_id].x + int(0.5*objs[local_id].width),
            objs[local_id].y + int(0.5*objs[local_id].height),
            area_max);

    return jint(cmd);
}
};