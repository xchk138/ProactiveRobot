from extract_digits import GetDashboardReader, PadSquare
from inference import YoloDetector, LABEL_NAMES
from utils import GetImages, Preprocess
import cv2

if __name__ == '__main__':
    ptr_color = (0,0,255)
    dsp_color = (0,255,0)
    board_color = (255,0,0)
    relax_ratio = 0.2
    images = GetImages('../data/samples/real/1')
    detector = YoloDetector('../models/yolov5n-pointer-1.onnx', num_class=3, score_thres=0.3, conf_thres=0.3)
    for id, fn, im_raw in images:
        if id < 5:
            continue
        im = Preprocess(im_raw)
        im_pad,_,_,_ = PadSquare(im, 640, 0)
        cv2.imshow('original', im_pad)
        cv2.waitKey(0)
        bboxes, labels, objs = detector.infer_closer(im, debug=False)
        for ibb in range(len(bboxes)):
            if LABEL_NAMES[labels[ibb]].lower() in ['dashboard']:
                x,y,w,h=bboxes[ibb]
                x -= w*relax_ratio/2
                y -= h*relax_ratio/2
                w *= (1.0+relax_ratio)
                h *= (1.0+relax_ratio)
                im_crop = im[int(y):int(y+h),int(x):int(x+w)]
                # find related pointers
                ptrs = [] # indices of pointer bounding boxes
                for ilb in range(len(labels)):
                    if LABEL_NAMES[labels[ilb]].lower() in ['pointer'] and objs[ilb]==objs[ibb]:
                        ptrs += [ilb]
                # for each pointer, transform the coordinate
                vis = cv2.resize(im_crop, (640,640))
                ratio_x = vis.shape[1] / im_crop.shape[1]
                ratio_y = vis.shape[0] / im_crop.shape[0]
                ptr_bboxes = []
                px, py = 30, 30
                for ptr_id in ptrs:
                    ptr_x = max(0, bboxes[ptr_id][0] - x)
                    ptr_y = max(0, bboxes[ptr_id][1] - y)
                    ptr_w = bboxes[ptr_id][2]
                    ptr_h = bboxes[ptr_id][3]
                    ptr_bboxes += [(ptr_x, ptr_y, ptr_w, ptr_h)]
                    cv2.rectangle(
                        vis, 
                        (int(ptr_x*ratio_x), int(ptr_y*ratio_y)), 
                        (int((ptr_x+ptr_w)*ratio_x), int((ptr_y+ptr_h)*ratio_y)),
                        ptr_color, 
                        2)
                    read_funcs = GetDashboardReader(im_crop, debug=True)
                    for func_id in range(len(read_funcs)):
                        if read_funcs[func_id].ready():
                            _read = read_funcs[func_id](ptr_bboxes[-1])
                            cv2.putText(vis, 'pointer# %d at ring#%d reads: %.1f' % (ptr_id, func_id, _read), (px, py), cv2.FONT_HERSHEY_PLAIN, 2.0, ptr_color,2)
                            py += 50
                cv2.imshow('crop#%d' % ibb, vis)
                cv2.waitKey(0)
