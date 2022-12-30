import paddleocr
print(paddleocr.__version__)
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
print(np.__version__)
import cv2
print(cv2.__version__)

def Float2Int(x):
    return tuple([int(e) for e in x])


def RotateImage(im, theta, scale, pad_val=0):
    affine_matrix = cv2.getRotationMatrix2D((im.shape[1]/2.0, im.shape[0]/2.0), theta, scale=scale)
    return cv2.warpAffine(im, affine_matrix, (im.shape[1], im.shape[0]), borderValue=(pad_val,pad_val,pad_val) if len(im.shape)==3 else pad_val)

def PadSquare(im, size, pad_val=0):
    h, w = im.shape[:2]
    pad_w, pad_h = 0, 0
    if h > w:
        pad_w = h - w
    else:
        pad_h = w - h
    if len(im.shape)==3:
        im_pad = np.zeros([h+pad_h, w+pad_w, im.shape[2]], im.dtype) + pad_val
        im_pad[pad_h//2:pad_h//2+h, pad_w//2:pad_w//2+w,:] = im[:,:,:]
    else:
        im_pad = np.zeros([h+pad_h, w+pad_w], im.dtype) + pad_val
        im_pad[pad_h//2:pad_h//2+h, pad_w//2:pad_w//2+w] = im[:,:]
    return cv2.resize(im_pad, (size, size), interpolation=cv2.INTER_LINEAR_EXACT)

def EnhanceContrast(im:np.ndarray)->np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    if len(im.shape)==3:
        h, w, c = im.shape
        if c==4:
            im = im[:,:,:3]
        # make both width and height even numbers
        new_w, new_h = w//2 *2, h//2 *2
        if new_w < w or new_h < h:
            im = im[:new_h, :new_w]
        im = cv2.cvtColor(im, cv2.COLOR_BGR2YUV_I420)
        im[:h,:] = clahe.apply(im[:h,:])
        im = cv2.cvtColor(im, cv2.COLOR_YUV2BGR_I420)
    else:
        im = clahe.apply(im)
    return im


def Smooth(im:np.ndarray)->np.ndarray:
    return cv2.medianBlur(im, 5)


def Binarize(im:np.ndarray)->np.ndarray:
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return 255 - cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize=9, C=20)

def WrapAffine(pts:np.ndarray, trans:np.ndarray):
    pts_new = []
    assert trans.shape[0] == 2 and trans.shape[1] == 3
    for pt in pts:
        x = trans[0,0]*pt[0] + trans[0,1]*pt[1] + trans[0,2]
        y = trans[1,0]*pt[0] + trans[1,1]*pt[1] + trans[1,2]
        pts_new.append((x,y))
    return pts_new

if __name__ == '__main__':
    tta_splits = 2
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
    # preprocess the image
    im = PadSquare(im, 640, pad_val=0)
    im = EnhanceContrast(im)
    im = Binarize(im)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    # test time augmentation
    scale = 0.9
    res = []
    angles = []
    
    for i in range(tta_splits):
        angles.append(i*90/tta_splits)
        res.append(ocr.ocr(RotateImage(im, angles[-1], scale))[0])
        if i > 0:
            angles.append(-i*90/tta_splits)
            res.append(ocr.ocr(RotateImage(im, angles[-1], scale))[0])

    # draw results
    vis = im.copy()
    _color  = (0,100,200)
    rec_thresh = 0.9

    bboxes = []
    values = []

    for i in range(len(res)):
        trans = cv2.getRotationMatrix2D((im.shape[1]/2.0,im.shape[0]/2.0), -angles[i], 1.0/scale)
        #print(trans)
        for pts, text in res[i]:
            if text[1] > rec_thresh:
                print(text)
                # convert bboxes back to original coordinate
                pts = WrapAffine(pts, trans)
                last_pt = pts[-1]
                for pt in pts:
                    cv2.line(vis, Float2Int(last_pt), Float2Int(pt), _color, 2, 8)
                    last_pt = pt
    
    cv2.imshow('ocr', vis)
    cv2.waitKey(0)