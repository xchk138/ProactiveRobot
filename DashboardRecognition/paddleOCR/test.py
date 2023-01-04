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

def RectInter(bb1, bb2):
    bx1 = max(bb1[0], bb2[0])
    by1 = max(bb1[1], bb2[1])
    bx2 = min(bb1[0] + bb1[2], bb2[0] + bb2[2])
    by2 = min(bb1[1] + bb1[3], bb2[1] + bb2[3])
    if bx2 > bx1 and by2 > by1:
        return (bx1, by1, bx2-bx1, by2-by1)
    else:
        return None

def RectUnion(bb1, bb2):
    bx1 = min(bb1[0], bb2[0])
    by1 = min(bb1[1], bb2[1])
    bx2 = max(bb1[0] + bb1[2], bb2[0] + bb2[2])
    by2 = max(bb1[1] + bb1[3], bb2[1] + bb2[3])
    return (bx1, by1, bx2-bx1, by2-by1)

def RectArea(bb):
    return bb[2]*bb[3]

def RectSmaller(bb1, bb2):
    if RectArea(bb1) > RectArea(bb2):
        return bb2
    else:
        return bb1

def RectBigger(bb1, bb2):
    if RectArea(bb1) > RectArea(bb2):
        return bb1
    else:
        return bb2

def RectIou(bb1, bb2):
    _overlap = RectInter(bb1, bb2)
    if _overlap is None:
        return 0
    else:
        return RectArea(_overlap) *1.0 / RectArea(RectSmaller(bb1, bb2))

def RectMerge(bboxes, max_iou=0.3)->list:
    is_removed = [False]*len(bboxes)
    for i in range(len(bboxes)):
        if is_removed[i]:
            continue
        for j in range(i+1, len(bboxes)):
            if is_removed[j]:
                continue
            if RectIou(bboxes[i], bboxes[j]) >= max_iou:
                # mark the smaller one as removed
                if RectArea(bboxes[i]) > RectArea(bboxes[j]):
                    is_removed[j] = True
                else:
                    is_removed[i] = True
    return is_removed

def Kmeans(points:list, group=2, max_iter=1000) -> list:
    # check group setting
    if group < 2:
        return [points]
    # check size of points, if not enough for clustering, then directly return
    if len(points) <= group:
        return [[pt] for pt in points]
    # choose 2 different points for cluster centers
    c = [points[0]]
    for i in range(1, group):
        all_same = True
        for j in range(1, len(points)):
            if points[j] not in c:
                all_same = False
                c += [points[j]]
                break
        if all_same:
            break
        

    
def TEST_EXPECT(r, v):
    if r != v:
        print('Test failed!')
    else:
        print('Test Okay')

def TEST_Kmeans():
    # test with single point in set
    ts = []
    res = Kmeans(ts, 2)
    TEST_EXPECT(res, [])
    ts = [1,2,3]
    res = Kmeans(ts, 1)
    TEST_EXPECT(res, [[1,2,3]])
    ts = [1]
    res = Kmeans(ts, 2)
    TEST_EXPECT(res, [[1]])
    ts = [1, 2]
    res = Kmeans(ts, 2)
    TEST_EXPECT(res, [[1],[2]])
    ts = [1, 2, 5]
    res = Kmeans(ts, 2)
    TEST_EXPECT(res, [[1,2],[5]])
    ts = [1,2, 5,7,9]
    res = Kmeans(ts, 2)
    TEST_EXPECT(res, [[1,2],[5,7,9]])

if __name__ == '__main__':

    # running test cases
    TEST_Kmeans()
    exit(0)

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

    quads = []
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
            quads.append(pts)
            values.append(val)
            last_pt = pts[-1]
            for pt in pts:
                cv2.line(vis, Float2Int(last_pt), Float2Int(pt), _color, 2, 8)
                last_pt = pt
    cv2.imshow('ocr1', vis)
    cv2.waitKey(0)

    # convert from quad to rect
    bboxes = []
    for pts in quads:
        min_x = min(min(pts[0][0], pts[1][0]), min(pts[2][0], pts[3][0]))
        max_x = max(max(pts[0][0], pts[1][0]), max(pts[2][0], pts[3][0]))
        min_y = min(min(pts[0][1], pts[1][1]), min(pts[2][1], pts[3][1]))
        max_y = max(max(pts[0][1], pts[1][1]), max(pts[2][1], pts[3][1]))
        bboxes.append((min_x, min_y, max_x-min_x, max_y-min_y))
    
    # remove duplicates or subparts
    bboxes_1 = []
    values_1 = []
    vis = im.copy()
    rm_flags = RectMerge(bboxes, 0.5)
    print('removed: %d / %d' % (np.sum(rm_flags), len(rm_flags)))
    for i in range(len(rm_flags)):
        if not rm_flags[i]:
            bboxes_1.append(bboxes[i])
            values_1.append(values[i])
            pts = (bboxes[i][0],bboxes[i][1],bboxes[i][0]+bboxes[i][2],bboxes[i][1]+bboxes[i][3])
            cv2.rectangle(vis, Float2Int(pts[:2]), Float2Int(pts[2:]), _color, 2, 8)
    cv2.imshow('ocr2', vis)
    cv2.waitKey(0)

    # solve the center coordinates
    # calculate the centers for each boxes
    centers = []
    for i in range(len(bboxes_1)):
        centers += [(bboxes_1[i][0] + bboxes_1[i][2]/2.0, bboxes_1[i][1] + bboxes_1[i][3]/2.0)]
    # param (x0,y0,r), for all (x,y), minimize sum of |((x,y) - (x0,y0))^2 - r^2|
    # consider 2 groups of bboxes
    # using KMeans algorithm to solve two cluster settings
    # given 1 center and 2 radius for 2 clusters
    # for case of single cluster, check distance between the 2 radiuses
    # if ratio of distance is under given threshold, then merge 2 clusters
    x0 = 0
    y0 = 0
    r1 = 0
    r2 = 0
    # mean x and mean y as the initial center
    for i in range(len(centers)):
        x0 += centers[i][0]
        y0 += centers[i][1]
    x0 /= len(centers)
    y0 /= len(centers)
    # calculate all distances
    dis = []
    for i in range(len(centers)):
        dis += [np.sqrt((centers[i][0] - x0)^2 + (centers[i][1] - y0)^2)]
    # clusterize the set of distance into 2 groups using Kmeans(Estimate-Minimizing)
    sets = Kmeans(dis, group=2, max_iter=1000)
    # for each group, minimizing minimize sum of |((x,y) - (x0,y0))^2 - r^2| 
    # to solve (x0,y0,r) for each cluster
    # then using new param to regroup all points in set
    
