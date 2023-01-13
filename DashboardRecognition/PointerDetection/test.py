from extract_digits import GetDashboardReader, PadSquare, EnhanceContrast, Binarize
from inference import YoloDetector, LABEL_NAMES
from utils import GetImages, Preprocess, LocatePointer
import cv2
from paddleocr import PaddleOCR

if __name__ == '__main__':
    ptr_color = (0,0,255)
    dsp_color = (0,255,0)
    board_color = (255,0,0)
    relax_ratio = 0.2
    model_dir = './pretrained/'
    #images = GetImages('../data/samples/real/1')
    images = GetImages('./')
    detector = YoloDetector(model_dir + '/yolo/yolov5n-pointer-1.onnx', num_class=3, score_thres=0.5, conf_thres=0.5)
    for id, fn, im_raw in images:
        if id < 3:
            continue
        print(fn)
        im = Preprocess(im_raw)
        im_h, im_w = im.shape[0:2]
        im_pad,pad_w,pad_h,pad_scale = PadSquare(im, 640)
        bboxes, labels, objs = detector.infer_closer(im, debug=False)
        for ibb in range(len(bboxes)):
            if LABEL_NAMES[labels[ibb]].lower() in ['dashboard']:
                x,y,w,h=bboxes[ibb]
                x1 = max(0, x-w*relax_ratio/2)
                y1 = max(0, y-h*relax_ratio/2)
                x2 = min(im_w, x+w+w*relax_ratio/2)
                y2 = min(im_h, y+h+h*relax_ratio/2)
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1
                # visualize the detection result
                cv2.rectangle(
                    im_pad, 
                    (int((x+pad_w/2)*pad_scale),int((y+pad_h/2)*pad_scale)), 
                    (int((x+w+pad_w/2)*pad_scale),int((y+h+pad_h/2)*pad_scale)), 
                    board_color, 
                    2)
                cv2.imshow('original', im_pad)
                cv2.waitKey(0)
                im_crop = im[int(y):int(y+h),int(x):int(x+w)]
                # find related pointers
                ptrs = [] # indices of pointer bounding boxes
                for ilb in range(len(labels)):
                    if LABEL_NAMES[labels[ilb]].lower() in ['pointer'] and objs[ilb]==objs[ibb]:
                        ptrs += [ilb]
                # for each pointer, transform the coordinate
                vis = cv2.resize(im_crop, (640,640))
                ratio_x = vis.shape[1] * 1.0 / im_crop.shape[1]
                ratio_y = vis.shape[0] * 1.0 / im_crop.shape[0]
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
                    read_funcs = GetDashboardReader(im_crop, debug=True, ocr_model_dir=model_dir)
                    for func_id in range(len(read_funcs)):
                        if read_funcs[func_id].ready():
                            _read = read_funcs[func_id](
                                ptr_bboxes[-1], 
                                LocatePointer(im_crop[int(ptr_y):int(ptr_y+ptr_h), int(ptr_x):int(ptr_x + ptr_w)], True)
                            )
                            cv2.putText(vis, 'pointer# %d at ring#%d reads: %.1f' % (ptr_id, func_id, _read), (px, py), cv2.FONT_HERSHEY_PLAIN, 2.0, ptr_color,2)
                            py += 50
                cv2.imshow('crop#%d' % ibb, vis)
                cv2.waitKey(0)
            elif LABEL_NAMES[labels[ibb]].lower() in ['display']:
                print('display found')
                ocr = PaddleOCR(
                    lang="en",
                    det_model_dir=model_dir + '/det/en_PP-OCRv3_det_infer',
                    rec_model_dir=model_dir + '/rec/en_PP-OCRv3_rec_infer',
                    cls_model_dir=model_dir + '/cls/ch_ppocr_mobile_v2.0_cls_infer',
                    use_angle_cls=True,
                    det_db_box_thresh=0.6,
                    cls_thresh=0.9,
                    det_db_thresh=0.3,
                    det_box_type='quad',
                    show_log=False)
                x,y,w,h=bboxes[ibb]
                x1 = max(0, x-w*relax_ratio/2)
                y1 = max(0, y-h*relax_ratio/2)
                x2 = min(im.shape[1], x+w/2+w*relax_ratio/2)
                y2 = min(im.shape[0], y+h/2+h*relax_ratio/2)
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1
                # visualize the detection result
                cv2.rectangle(
                    im_pad, 
                    (int((x+pad_w/2)*pad_scale),int((y+pad_h/2)*pad_scale)), 
                    (int((x+w+pad_w/2)*pad_scale),int((y+h+pad_h/2)*pad_scale)), 
                    board_color, 
                    2)
                cv2.imshow('original', im_pad)
                cv2.waitKey(0)
                im_crop = im[int(y):int(y+h),int(x):int(x+w)]
                im_crop, im_pad_w,im_pad_h,im_scale = PadSquare(im_crop, 640, pad_val=0)
                im_crop = EnhanceContrast(im_crop)
                im_crop = Binarize(im_crop)
                im_crop = cv2.cvtColor(im_crop, cv2.COLOR_GRAY2BGR)
                res = ocr.ocr(im_crop, det=False, rec=True,cls=False)[0]
                print(res)