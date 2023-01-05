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

def norm_L1(d):
            return np.abs(d)
def calc_dist(p, c):
    return np.sum(norm_L1(p-c))

def Kmeans(points:list, group=2, min_move=1.0, max_iter=30) -> list:
    # check group setting
    if group < 2:
        return [[i for i in range(len(points))]]
    # check size of points, if not enough for clustering, then directly return
    if len(points) <= group:
        return [[i] for i in range(len(points))]
    # choose different points for cluster centers
    centers = [points[0]]
    for i in range(1, group):
        all_same = True
        for j in range(1, len(points)):
            if points[j] not in centers:
                all_same = False
                centers += [points[j]]
                break
        if all_same:
            group = i
            break
    
    # iteration begin
    centers = np.array(centers)
    centers_last = centers.copy()
    for itr in range(max_iter):
        # update clusters using centers 
        assert len(centers) == group
        clusters = [[] for _ in range(group)] # store indexes of points
        for i in range(len(points)):
            dists = [calc_dist(points[i], c) for c in centers]
            id = np.argmin(dists)
            clusters[id] += [i]
        # the EM(expectation-maximization algorithm)
        # update centers of clusters
        assert len(clusters) == len(centers)
        for i in range(len(clusters)):
            c_new = None
            if type(points[0]) == int or type(points[0])==float: # points are scalars
                c_new = 0
            else: # points are vectors
                c_new = np.zeros_like(points[0])
            for id in clusters[i]:
                c_new = c_new + np.array(points[id])
            c_new = c_new / len(clusters[i])
            centers[i] = c_new
        # check stop condition: center movement 
        _move = np.max(np.sqrt(np.sum(np.square(centers_last - centers), axis=-1)))
        print('kmeans itr# %d with center move: %.3f' % (itr, _move))
        if _move <  min_move:
            print('kmeans converged')
            break
        centers_last = centers.copy()
    return clusters
    
def EXPECT_EQ(r, e, tname): # r is real value, e is expected value
    if r != e:
        print('[%s] Test failed' % tname)
    else:
        print('[%s] Test pass' % tname)

def EXPECT_EQ_SET(r, e, tname): # r is real set, e is expected set
    for rr in r:
        if rr not in e:
            print('[%s] Test failed' % tname)
    for ee in e:
        if ee not in r:
            print('[%s] Test failed' % tname)
    print('[%s] Test pass' % tname)

def EXPECT_IN(r, e, tname):
    for ee in e:
        if r == ee:
            print('[%s] Test pass' % tname)
            break
    print('[%s] Test failed' % tname)

def TEST_Kmeans():
    # test with single point in set
    ts = []
    res = Kmeans(ts, 2)
    EXPECT_EQ(res, [], 'T1')
    ts = [1,2,3]
    res = Kmeans(ts, 1)
    EXPECT_EQ(res, [[0,1,2]], 'T2')
    ts = [1]
    res = Kmeans(ts, 2)
    EXPECT_EQ(res, [[0]], 'T3')
    ts = [1, 2]
    res = Kmeans(ts, 2)
    EXPECT_EQ_SET(res, [[0],[1]], 'T4')
    ts = [1.2, 2.3, 5.6]
    res = Kmeans(ts, 2)
    EXPECT_EQ_SET(res, [[0,1],[2]], 'T5')
    ts = [7,5,1,2,9]
    res = Kmeans(ts, 2)
    EXPECT_EQ_SET(res, [[2,3],[0,1,4]], 'T6')

if __name__ == '__main__':
    _TEST_MODE_ = False
    if _TEST_MODE_:
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
        show_log=False)

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
    pts = []
    for i in range(len(bboxes_1)):
        pts += [(bboxes_1[i][0] + bboxes_1[i][2]/2.0, bboxes_1[i][1] + bboxes_1[i][3]/2.0)]
    # param (x0,y0,r), for all (x,y), minimize sum of |((x,y) - (x0,y0))^2 - r^2|
    # consider 2 groups of bboxes
    # using KMeans algorithm to solve two cluster settings
    # given 1 center and 2 radius for 2 clusters
    # for case of single cluster, check distance between the 2 radiuses
    # if ratio of distance is under given threshold, then merge 2 clusters
    x0 = 0
    y0 = 0
    # mean x and mean y as the initial center
    for i in range(len(pts)):
        x0 += pts[i][0]
        y0 += pts[i][1]
    x0 /= len(pts)
    y0 /= len(pts)
    centers = [(x0, y0)]
    # iterate to solve center and radius
    MAX_ITER = 30
    for itr in range(MAX_ITER):
        # calculate all distances
        dis = []
        for i in range(len(pts)):
            dis += [np.sqrt((pts[i][0] - centers[0][0])*(pts[i][0] - centers[0][0]) + (pts[i][1] - centers[0][1])*(pts[i][1] - centers[0][1]))]
        # clusterize the set of distance into 2 groups using Kmeans(Estimate-Minimizing)
        clusters = Kmeans(dis, group=2, max_iter=30) # max cluster number is set to $group
        print(clusters)
        # for each group, minimizing sum of |((x,y) - (x0,y0))^2 - r^2| 
        # to solve this minimization problem, we convert it to linear solution as follow:
        # W = (A^T*A)^{-1}*A^T*C
        # where, W = [x_0, y_0, r_0^2-x_0^2-y_0^2]^T, to be solved thru matrix ops
        # A = [[2x_1,2y_1,1],[2x_2,2y_2,1],[2x_3,2y_3,1],...]
        # C = [x_1^2+y_1^2,x_2^2+y_2^2,...]
        # where, <x_i, y_i> are sampled points of <x,y> from cluster.
        # to solve (x0,y0,r) for each cluster, we apply such procedures,
        centers_last = np.array(centers)
        centers = []
        rads = []
        pts = np.array(pts)
        vis = im.copy()
        for clu in clusters:
            v_x = pts[clu,0]
            v_y = pts[clu,1]
            A = np.stack([2.0*v_x, 2.0*v_y, np.ones_like(v_x)], axis=0).T
            C = (v_x*v_x + v_y*v_y).T
            _st, iATA = cv2.invert(A.T@A)
            assert _st > 0
            # iATA = np.linalg.inv(A.T@A)
            W = iATA @ A.T @ C
            x0 = W[0]
            y0 = W[1]
            assert W[2] + (x0*x0 + y0*y0) > 0
            r0 = np.sqrt(W[2] + (x0*x0 + y0*y0))
            cv2.circle(vis, (int(x0),int(y0)), int(r0), _color, 2)
            rads += [r0]
            centers += [(x0, y0)]
        cv2.imshow('centers solved', vis)
        cv2.waitKey(0)
        # check if two radius are close enough, if yes then merge them
        if len(rads) == 2 and np.abs(rads[0] - rads[1]) / np.max(rads) < 0.18:
            rads = [(rads[0] + rads[1])/2.0]
            clusters = [clusters[0] + clusters[1]]
        # check if center is stable to prove converge
        centers = np.array(centers)
        centers = np.array([(centers[0] + centers[1])/2.0])
        c_mov = np.sqrt(np.sum(np.square(centers - centers_last)))
        print('EM itr#%d with center shift: %.3f' % (itr, c_mov))
        if c_mov < np.max(rads) * 0.05:
            print('center movement converged!')
            break
    print('final cluster result:')
    print('center: (%d, %d)' % (int(centers[0][0]), int(centers[0][1])))
    print('cluster:' + str(clusters))
    # visualize the boxes for each board-tick axis
    vis = im.copy()
    colors = [(0,0,255), (255, 0, 0), (0, 255, 0)]
    cv2.circle(vis, (int(x0),int(y0)), 30, colors[-1], 2)
    for k in range(len(clusters)):
        for i in clusters[k]:
            pts = (bboxes_1[i][0],bboxes_1[i][1],bboxes_1[i][0]+bboxes_1[i][2],bboxes_1[i][1]+bboxes_1[i][3])
            cv2.rectangle(vis, Float2Int(pts[:2]), Float2Int(pts[2:]), colors[k], 2, 8)
    cv2.imshow('cluster result', vis)
    cv2.waitKey(0)