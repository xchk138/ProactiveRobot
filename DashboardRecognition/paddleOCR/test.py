import paddleocr
print(paddleocr.__version__)
from paddleocr import PaddleOCR, draw_ocr
import time
import cv2
print(cv2.__version__)

def Float2Int(x):
    return tuple([int(e) for e in x])


def RotateImage(im, theta, scale):
    affine_matrix = cv2.getRotationMatrix2D((im.shape[1]/2.0, im.shape[0]/2.0), theta, scale=scale)
    return cv2.warpAffine(im, affine_matrix, (im.shape[1], im.shape[0]))

if __name__ == '__main__':
    # test image
    im_path = "t1.png"
    ocr = PaddleOCR(
        lang="en",
        det_model_dir='pretrained/det/en_PP-OCRv3_det_infer',
        rec_model_dir='pretrained/rec/en_PP-OCRv3_rec_infer',
        cls_model_dir='pretrained/cls/ch_ppocr_mobile_v2.0_cls_infer',
        use_angle_cls=True,
        det_db_box_thresh=0.6,
        cls_thresh=0.9,
        det_db_thresh=0.3,
        det_box_type='quad',
        show_log=False
        )

    im = cv2.imread(im_path)
    im = cv2.resize(im, (640,640))
    # preprocess the image
    im = RotateImage(im, -60, 1.0)
    res = ocr.ocr(im)

    # draw results
    vis = im.copy()
    _color  = (0,100,200)
    rec_thresh = 0.7

    angles = []
    values = []

    for pts, text in res[0]:
        if text[1] > rec_thresh:
            print(text)
            last_pt = pts[-1]
            for pt in pts:
                cv2.line(vis, Float2Int(last_pt), Float2Int(pt), _color, 3, 8)
                last_pt = pt
    
    cv2.imshow('ocr', vis)
    cv2.waitKey(0)