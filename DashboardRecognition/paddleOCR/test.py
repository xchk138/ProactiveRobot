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


def IsDigit(v:str)->bool:
    c_ = ord(v) 
    return c_ >= ord('0') and c_ <= ord('9')

def IsInteger(v:str)-> bool:
    if v == '0':
        return True
    elif v[:1] == '0':
        return False
    for i in range(len(v)):
        chr = v[i:i+1]
        if not IsDigit(chr):
            return False
    return True

def SegDecimal(v:str)-> list:
    pid = v.find('.', 0, len(v))
    if pid >= 0:
        return [v[:pid],v[pid+1:]]
    else:
        return [v]

def IsFloat(v:str)->bool:
    for i in range(len(v)):
        chr = v[i:i+1]
        if not IsDigit(chr):
            return False
    if v[len(v)-1:] == '0':
        return False
    return True


def IsDecimal(v:str):
    parts = SegDecimal(v)
    if len(parts) == 1:
        return IsInteger(v)
    else:
        return IsInteger(parts[0]) and IsFloat(parts[1])

def GetDecimal(v:str) -> float:
    assert IsDecimal(v)
    parts = SegDecimal(v)
    if len(parts) == 1:
        return int(v) # atoi_s(v)
    else:
        return float(v) # atof_s(v)


if __name__ == '__main__':
    tta_splits = 3
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

    # extract all qualified recognition results
    for i in range(len(res)):
        trans = cv2.getRotationMatrix2D((im.shape[1]/2.0,im.shape[0]/2.0), -angles[i], 1.0/scale)
        #print(trans)
        for pts, text in res[i]:
            conf = text[1]
            if conf < rec_thresh:
                continue
            if not IsDecimal(text[0]):
                continue
            val = GetDecimal(text[0])
            print('val: %.3f conf: %.3f' % (val, conf))
            # convert bboxes back to original coordinate
            pts = WrapAffine(pts, trans)
            # add board-ticks
            bboxes.append(pts)
            values.append(val)
            last_pt = pts[-1]
            for pt in pts:
                cv2.line(vis, Float2Int(last_pt), Float2Int(pt), _color, 2, 8)
                last_pt = pt
    # remove duplicates or subparts
    bboxes_1 = []
    values_1 = []
    if i in range(len(values)):
        if bboxes_1[]
        bboxes_1.append(bboxes[i])
    
    cv2.imshow('ocr', vis)
    cv2.waitKey(0)