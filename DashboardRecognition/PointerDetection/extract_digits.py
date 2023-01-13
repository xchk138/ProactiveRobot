import paddleocr
print('paddleocr: ' + paddleocr.__version__)
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
print('numpy: ' + np.__version__)
import cv2
print('opencv-python: ' + cv2.__version__)

# turn on to enable testing modules
_TEST_MODE_ = True
_DEBUG_MODE_ = True

def CompSort(x:list, sort_dim=0, ascending=False)->list:
    assert type(x) == list
    if type(x[0]) == tuple:
        v = [(i, x[i][sort_dim]) for i in range(len(x))]
    else:
        v = enumerate(x)
    comp = sorted(v, key=lambda e:e[1], reverse=ascending)
    map_ids = [comp[i][0] for i in range(len(comp))]
    y = [x[id] for id in map_ids]
    return y

class StagedLinearFunction(object):
    def __init__(
        self, 
        x:np.ndarray=None, 
        y:np.ndarray=None, 
        center=(0,0), 
        radius=0, 
        origin=0,
        debug=False) -> None:
        self.available = True
        self.debug = debug
        self.center = center
        self.radius = radius
        self.origin = origin
        if x is None or y is None:
            if debug:
                print('input x or y is none!')
            self.available = False
            return
        if len(x) != len(y):
            if debug:
                print('dim of x and y are different!')
            self.available = False
            return
        if len(x) < 2:
            if debug:
                print('sample volume is insufficient!')
            self.available = False
            return
        # model is Y = kX + b
        # X is angle, Y is the value recognized(may be wrong)
        # calculating the linearity coefficient R,
        # check if linear model is satisfied, otherwise using staged linear model
        # variable x and y
        if type(x) == list:
            var_x = np.array(x)
        else:
            var_x = x
        if type(y) == list:
            var_y = np.array(y)
        else:
            var_y = y
        # get average of variable x
        mean_x = var_x.mean()
        mean_y = var_y.mean()
        # cross variance of X and Y
        var_XY = np.sum((var_x - mean_x)*(var_y - mean_y))
        # self variance of X
        var_XX = np.sum((var_x - mean_x)*(var_x - mean_x))
        # self variance of Y
        var_YY = np.sum((var_y - mean_y)*(var_y - mean_y))
        # calculate linear coefficient
        if var_XX < 1e-6 or var_YY < 1e-6:
            if debug:
                print('self variance of samples check failed!')
            self.available = False
            return
        lin_coef = var_XY / np.sqrt(var_XX*var_YY)
        if debug:
            print('Linear coefficient: %.3f' % lin_coef)
        if abs(lin_coef) >= 0.96:
            self.num_stages = 0
            x = sorted(x, reverse=False) # ascending order
            self.stages = [x[0], x[-1]]
            # calculating the parameter for linear equation y=Bx+A
            pB = var_XY / var_XX
            pA = mean_y - mean_x * pB
            self.params = [(pA, pB)]
        else:
            self.num_stages = 0
            self.params = []
            # merge all variables into samples
            samples = [(x[id], y[id]) for id in range(len(x))]
            # sort according to the specified dimension: angles
            samples_sorted = CompSort(samples, sort_dim=0)
            # split sorted samples into sorted variables
            x = [samples_sorted[id][0] for id in range(len(samples_sorted))]
            y = [samples_sorted[id][1] for id in range(len(samples_sorted))]
            self.stages = [x[0]]
            last_x = x[0]
            last_y = y[0]
            for i in range(1, len(x)):
                if x[i] - last_x > 1e-5:
                    pB = (y[i] - last_y) / (x[i] - last_x) 
                    pA = last_y
                    self.params += [(pA, pB)]
                    last_x = x[i]
                    last_y = y[i]
                    self.stages += [x[i]]
                    self.num_stages += 1
            if debug:
                print('number of stages: %d' % self.num_stages)
                print('staged linear function params: ')
                print(self.params)
            if self.num_stages <= 0:
                self.available = False
                return
    def ready(self):
        return self.available
    # x is the bounding box of pointer, x,y,w,h
    # diagonal id is the diagonal line index of pointer,
    # diagonal_id=-1 stands for uncertainty of choice.
    # diagonal_id=0 stands for diagonal line from top left to bottom right
    # diagonal_id=1 stands for diagonal line from top right to bottom left
    def __call__(self, x:tuple, diagonal_id=-1):
        if not self.available:
            if self.debug:
                print('function not available!')
            return -1
        if type(x) != tuple and type(x) != list:
            if self.debug:
                print('bad call: input requires bounding box as tuple or list!')
            return -1
        if len(x) != 4:
            if self.debug:
                print('bad call: input requires point of 4 dim as (x,y,w,h)!')
            return -1
        # get the pointer end
        ptr_end = (0,0)
        if diagonal_id==-1:
            # convert from bbox into quad pointers
            pts = [(x[0],x[1]), (x[0]+x[2],x[1]), (x[0]+x[2], x[1]+x[3]), (x[0],x[1]+x[3])]
            # find the farest point, it's the pointer end
            dis = []
            if self.debug:
                print('the board center: ' + str(self.center))
            for pt in pts:
                dis += [np.sum(np.square(np.array(pt) - np.array(self.center)))]
            max_id = np.argmax(dis)
            if self.debug:
                if max_id == 0:
                    print('the left top is the end of pointer')
                elif max_id == 1:
                    print('the right top is the end of pointer')
                elif max_id == 2:
                    print('the right bottom is the end of pointer')
                else:
                    print('the left bottom is the end of pointer')
            ptr_end = pts[max_id]
        elif diagonal_id==0:
            # the diagonal line from TL to BR is the pointer
            # convert from bbox into points of TL and BR
            pts = [(x[0],x[1]), (x[0]+x[2], x[1]+x[3])]
            # find the farest point, it's the pointer end
            dis = []
            if self.debug:
                print('the board center: ' + str(self.center))
            for pt in pts:
                dis += [np.sum(np.square(np.array(pt) - np.array(self.center)))]
            max_id = np.argmax(dis)
            if self.debug:
                if max_id == 0:
                    print('the left top is the end of pointer')
                else:
                    print('the right bottom is the end of pointer')
            ptr_end = pts[max_id]
        elif diagonal_id==1:
            # the diagonal line from TR to BL is the pointer
            # convert from bbox into points of TL and BR
            pts = [(x[0]+x[2],x[1]),  (x[0],x[1]+x[3])]
            # find the farest point, it's the pointer end
            dis = []
            if self.debug:
                print('the board center: ' + str(self.center))
            for pt in pts:
                dis += [np.sum(np.square(np.array(pt) - np.array(self.center)))]
            max_id = np.argmax(dis)
            if self.debug:
                if max_id == 0:
                    print('the right top is the end of pointer')
                else:
                    print('the left bottom is the end of pointer')
            ptr_end = pts[max_id]
        else:
            print('diagonal_id is invalid!')
            assert False
        # get the angle
        vec = np.array(ptr_end) - np.array(self.center)
        dis = norm_L2(vec)
        ang = 0
        if dis > self.radius*0.3: # too close ignore this point
            vec = vec / dis # normalized vector is unit vector
            vec[1] = -vec[1]
            ang = np.arccos(vec[0])
            if vec[1] < 0: # y < 0
                ang = 2*np.pi - ang
        # transform angle coordinate
        ang = ang - self.origin
        x = ang
        # check if the angle exceeds the range
        if x < self.stages[0]:
            if self.stages[0] - x > abs(2*np.pi+x - self.stages[-1]):
                x += 2*np.pi
        if x > self.stages[-1]:
            if x - self.stages[-1] > abs(-2*np.pi+x - self.stages[0]):
                x -= 2*np.pi
        # calculate the value of given angle
        if self.num_stages > 0:
            stage_id = -1
            for i in range(len(self.stages)):
                if x < self.stages[i]:
                    break
                stage_id += 1
            if stage_id < 0: # range (-inf, stagess[0])
                if self.debug:
                    print('warning: call staged linear function out of range!')
                stage_id = 0
            elif stage_id >= self.num_stages: # range (stages[-1], +inf)
                if self.debug:
                    print('warning: call staged linear function out of range!')
                stage_id = self.num_stages - 1
            else:
                if self.debug:
                    print('hit stage: %d' % stage_id)
            # apply the linear model according to stage index
            return self.params[stage_id][0] + (x - self.stages[stage_id]) * self.params[stage_id][1]
        else: # single linear equation without stages
            return self.params[0][0] + x * self.params[0][1]

def Float2Int(x:list):
    return tuple([int(e) for e in x])

def RotateImage(im:np.ndarray, theta:float, scale:float, pad_val=0):
    affine_matrix = cv2.getRotationMatrix2D((im.shape[1]/2.0, im.shape[0]/2.0), theta, scale=scale)
    return cv2.warpAffine(im, affine_matrix, (im.shape[1], im.shape[0]), borderValue=(pad_val,pad_val,pad_val) if len(im.shape)==3 else pad_val)

def PadSquare(im:np.ndarray, size:int, pad_val=0)->np.ndarray:
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
    assert im_pad.shape[0] == im_pad.shape[1]
    scale = 1.0*size / im_pad.shape[0]
    return cv2.resize(im_pad, (size, size), interpolation=cv2.INTER_LINEAR_EXACT), pad_w, pad_h, scale

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
    return 255 - cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize=9, C=10) # C=15

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
    # elif v[:1] == '0':
    #     return False
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
    if v[0] == '-': # negative figure
        v = v[1:]
    parts = SegDecimal(v)
    if len(parts) == 1:
        return IsInteger(v)
    else:
        return IsInteger(parts[0]) and IsFloat(parts[1])

def GetDecimal(v:str) -> float:
    assert IsDecimal(v)
    neg_ = 1
    if v[0] == '-': # negative figure
        v = v[1:]
        neg_ = -1
    parts = SegDecimal(v)
    if len(parts) == 1:
        return neg_*int(v) # atoi_s(v)
    else:
        if v[0]=='0' and v[1]!='.': # the dot is missing, we put it back
            v = '.'.join(['0', v[1:]])
        return neg_*float(v) # atof_s(v)

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
    return np.mean(np.abs(d))

def norm_L2(d):
    return np.sqrt(np.sum(np.square(d)))

def calc_dist(p, c):
    return np.sum(np.abs(p-c))

def Kmeans(points:list, group=2, min_move=1.0, max_iter=30, debug=False) -> list:
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
        if debug:
            print('kmeans itr# %d with center move: %.3f' % (itr, _move))
        if _move <  min_move:
            if debug:
                print('kmeans converged')
            break
        centers_last = centers.copy()
    return clusters
    
def EXPECT_EQ(r, e, tname): # r is real value, e is expected value
    if r != e:
        print('[%s] Test failed' % tname)
        return False
    else:
        print('[%s] Test pass' % tname)
        return True

def EXPECT_EQ_SET(r, e, tname): # r is real set, e is expected set
    for rr in r:
        if rr not in e:
            print('[%s] Test failed' % tname)
            return False
    for ee in e:
        if ee not in r:
            print('[%s] Test failed' % tname)
            return False
    print('[%s] Test pass' % tname)
    return True

def EXPECT_IN(r, e, tname):
    for ee in e:
        if r == ee:
            print('[%s] Test pass' % tname)
            return True
    print('[%s] Test failed' % tname)
    return False

def TEST_Kmeans(debug=False):
    # test with single point in set
    ts = []
    res = Kmeans(ts, 2, debug=debug)
    EXPECT_EQ(res, [], 'T1')
    ts = [1,2,3]
    res = Kmeans(ts, 1, debug=debug)
    EXPECT_EQ(res, [[0,1,2]], 'T2')
    ts = [1]
    res = Kmeans(ts, 2, debug=debug)
    EXPECT_EQ(res, [[0]], 'T3')
    ts = [1, 2]
    res = Kmeans(ts, 2, debug=debug)
    EXPECT_EQ_SET(res, [[0],[1]], 'T4')
    ts = [1.2, 2.3, 5.6]
    res = Kmeans(ts, 2, debug=debug)
    EXPECT_EQ_SET(res, [[0,1],[2]], 'T5')
    ts = [7,5,1,2,9]
    res = Kmeans(ts, 2, debug=debug)
    EXPECT_EQ_SET(res, [[2,3],[0,1,4]], 'T6')

def RingShift(x:list, pos:int):
    if type(x) == np.ndarray:
        _x = x.tolist()
    else:
        _x = x
    _x = _x[pos:] + _x[:pos]
    if type(x) == np.ndarray:
        return np.array(_x)
    else:
        return _x

def CountDisorder(x:np.ndarray)->int:
    num_pos = 0
    #num_neg = 0
    if type(x) == list:
        _x = np.array(x)
    else:
        _x = x
    for i in range(1,len(x)):
        num_pos += np.sum((_x[i] - _x[:i]) > 0)
        #num_neg += np.sum((_x[i] - _x[:i]) < 0)
    #return min(num_pos, num_neg)
    return num_pos

def GetDashboardReader(
    im: np.ndarray, 
    tta_splits=3, 
    ocr_model_dir='pretrained/', 
    rec_thresh = 0.9,
    debug=False) -> list: # return list of [StagedLinearFunction]
    ocr = PaddleOCR(
        lang="en",
        det_model_dir=ocr_model_dir + '/det/en_PP-OCRv3_det_infer',
        rec_model_dir=ocr_model_dir + '/rec/en_PP-OCRv3_rec_infer',
        cls_model_dir=ocr_model_dir + '/cls/ch_ppocr_mobile_v2.0_cls_infer',
        use_angle_cls=True,
        det_db_box_thresh=0.6,
        cls_thresh=0.9,
        det_db_thresh=0.3,
        det_box_type='quad',
        show_log=False)
    # preprocess the image
    im, im_pad_w,im_pad_h,im_scale = PadSquare(im, 640, pad_val=0)
    #im = EnhanceContrast(im)
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
    quads = []
    values = []
    
    # extract all qualified recognition results
    for i in range(len(res)):
        trans = cv2.getRotationMatrix2D((im.shape[1]/2.0,im.shape[0]/2.0), -angles[i], 1.0/scale)
        #print(trans)
        for pts, text in res[i]:
            if debug:
                print('text: %s conf: %.3f' % (text[0], text[1]))
            conf = text[1]
            if conf < rec_thresh:
                continue
            if not IsDecimal(text[0]):
                continue
            val = GetDecimal(text[0])
            if debug:
                print('val: %.3f conf: %.3f' % (val, conf))
            # convert bboxes back to original coordinate
            pts = WrapAffine(pts, trans)
            # add board-ticks
            quads.append(pts)
            values.append(val)
            if debug:
                last_pt = pts[-1]
                for pt in pts:
                    cv2.line(vis, Float2Int(last_pt), Float2Int(pt), _color, 2, 8)
                    last_pt = pt
    if debug:
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
    if debug:
        vis = im.copy()
    rm_flags = RectMerge(bboxes, 0.5)
    if debug:
        print('removed: %d / %d' % (np.sum(rm_flags), len(rm_flags)))
    for i in range(len(rm_flags)):
        if not rm_flags[i]:
            bboxes_1.append(bboxes[i])
            values_1.append(values[i])
            if debug:
                pts = (bboxes[i][0],bboxes[i][1],bboxes[i][0]+bboxes[i][2],bboxes[i][1]+bboxes[i][3])
                cv2.rectangle(vis, Float2Int(pts[:2]), Float2Int(pts[2:]), _color, 2, 8)
    if debug:
        cv2.imshow('ocr2', vis)
        cv2.waitKey(0)
    # check if points are enough
    if len(bboxes_1) <= 3:
        return []

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
    # method #1
    # mean x and mean y as the initial center
    # this approach sometimes goes into a bad solution,sub-optimal
    USE_MEAN_AS_CENTER = False
    if USE_MEAN_AS_CENTER:
        for i in range(len(pts)):
            x0 += pts[i][0]
            y0 += pts[i][1]
        x0 /= len(pts)
        y0 /= len(pts)
        centers = [(x0, y0)]
    # method #2
    # set the center as the center of the image
    centers = [(im.shape[1]/2.0, im.shape[0]/2.0)]
    # iterate to solve center and radius
    MAX_ITER = 30
    for itr in range(MAX_ITER):
        # calculate all distances
        dis = []
        for i in range(len(pts)):
            dis += [np.sqrt((pts[i][0] - centers[0][0])*(pts[i][0] - centers[0][0]) + (pts[i][1] - centers[0][1])*(pts[i][1] - centers[0][1]))]
        # clusterize the set of distance into 2 groups using Kmeans(Estimate-Minimizing)
        clusters = []
        if len(pts) >= 3: # locate a center requires at least 3 points 
            clusters = Kmeans(dis, group=2, max_iter=30, debug=debug) # max cluster number is set to $group
        else:
            if debug:
                print('total points are insufficient, task canclled!')
            return []
        if debug:
            print(clusters)
        # remove cluster without enough points
        if len(clusters)==2 and (len(clusters[0]) < 3 or len(clusters[1]) < 3):
            if debug:
                print('points in one cluster are insufficient, cluster merged!')
            clusters = [clusters[0] + clusters[1]]
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
        if debug:
            vis = im.copy()
        clu_rm_flags = [False for _ in range(len(clusters))]
        for icl in range(len(clusters)):
            clu = clusters[icl]
            v_x = pts[clu,0]
            v_y = pts[clu,1]
            A = np.stack([2.0*v_x, 2.0*v_y, np.ones_like(v_x)], axis=0).T
            C = (v_x*v_x + v_y*v_y).T
            _st, iATA = cv2.invert(A.T@A)
            if _st <= 0:
                if debug:
                    print('inverse matrix dose not exist!')
                # mark this cluster as bad
                clu_rm_flags[icl] = True
                continue
            # iATA = np.linalg.inv(A.T@A)
            W = iATA @ A.T @ C
            x0 = W[0]
            y0 = W[1]
            if W[2] + (x0*x0 + y0*y0) <= 0:
                if debug:
                    print('radius of dashboard cannot be negative!')
                # mark this cluster as bad
                clu_rm_flags[icl] = True
                continue
            r0 = np.sqrt(W[2] + (x0*x0 + y0*y0))
            if debug:
                cv2.circle(vis, (int(x0),int(y0)), int(r0), _color, 2)
            rads += [r0]
            centers += [(x0, y0)]
        if debug:
            cv2.imshow('centers solved', vis)
            cv2.waitKey(0)
        # remove bad cluster in collection
        clusters_1 = []
        for icl in range(len(clusters)):
            if not clu_rm_flags[icl]:
                clusters_1.append(clusters[icl])
        clusters = clusters_1
        # check if two radius are close enough, if yes then merge them
        if len(rads) == 2 and np.abs(rads[0] - rads[1]) / np.max(rads) < 0.18:
            rads = [(rads[0] + rads[1])/2.0]
            clusters = [clusters[0] + clusters[1]]
        # check if center is stable to prove converge
        centers = np.array(centers)
        if len(centers) == 2:
            centers = np.array([(centers[0] + centers[1])/2.0])
        c_mov = np.sqrt(np.sum(np.square(centers - centers_last)))
        if debug:
            print('EM itr#%d with center shift: %.3f' % (itr, c_mov))
        if c_mov < np.max(rads) * 0.05:
            if debug:
                print('center movement converged!')
            break
    if debug:
        print('final cluster result:')
        print('center: (%d, %d)' % (int(centers[0][0]), int(centers[0][1])))
        print('cluster:' + str(clusters))
        print('values:')
        print(np.array(values_1)[clusters[0]])
        if len(clusters) > 1:
            print(np.array(values_1)[clusters[1]])
        # visualize the boxes for each board-tick axis
        vis = im.copy()
        colors = [(0,0,255), (255, 0, 0), (0, 255, 0)]
        cv2.circle(vis, (int(x0),int(y0)), 30, colors[-1], 2)
        for k in range(len(clusters)):
            for i in clusters[k]:
                _pts = (bboxes_1[i][0],bboxes_1[i][1],bboxes_1[i][0]+bboxes_1[i][2],bboxes_1[i][1]+bboxes_1[i][3])
                cv2.rectangle(vis, Float2Int(_pts[:2]), Float2Int(_pts[2:]), colors[k], 2, 8)
        cv2.imshow('cluster result', vis)
        cv2.waitKey(0)
    # checking the cluster distribution, if distances to the center has large variance,
    # then this cluster may contains bad points, we need remove them
    # calculate the average distance to the center to be radius of circle
    for k in range(len(clusters)):
        dis_var = 0
        for i in clusters[k]:
            _dis = np.sqrt(np.sum(np.square(np.array(pts[i]) - np.array(centers[0]))))
            dis_var += (_dis - rads[k])*(_dis - rads[k])
        dis_var = np.sqrt(dis_var/len(clusters[k]))
        if dis_var > 0.2*rads[k]:
            if debug:
                print('large variance found inside cluster#%d: %.3f' %(k, dis_var))
            # required to remove bad points inside cluster
            # using ransac algorithm to optimize the clustering process
            # randomly select 3 point to calculate the center of radius, 
            # then estimate the rest of points.
            # if over 90% or higher of the points support this solution,
            # then the solution is optimal.
            
    # =============== modeling angles and values ================
    # get angles
    angles = [[] for _ in range(len(clusters))]
    values = [[] for _ in range(len(clusters))]
    clusters_new = [[] for _ in range(len(clusters))]
    for i in range(len(clusters)):
        for id in clusters[i]:
            vec = np.array(pts[id]) - np.array(centers[0])
            dis = norm_L2(vec)
            if dis > rads[i]*0.3: # too close ignore this point
                clusters_new[i] += [id]
                vec = vec / dis # normalized vector is unit vector
                vec[1] = -vec[1]
                ang = np.arccos(vec[0])
                if vec[1] < 0: # y < 0
                    ang = 2*np.pi - ang
                angles[i] += [ang]
                values[i] += [values_1[id]]
    # sort angles, and calc gradients of values for every 2 pair points in neighborhood
    ret_funcs = []
    for i in range(len(clusters)):
        if debug:
            print('======Ring #%d======'%i)
        # merge all variables into samples
        samples = [(angles[i][id], values[i][id]) for id in range(len(angles[i]))]
        # sort according to the specified dimension: angles
        samples_sorted = CompSort(samples, 0)
        # split sorted samples into sorted variables
        _angs = [samples_sorted[id][0] for id in range(len(samples_sorted))] 
        _vals = [samples_sorted[id][1] for id in range(len(samples_sorted))]
        # find the origin of the board-ring
        # to minimize the disorder of values
        num_disorders = [CountDisorder(RingShift(_vals, _shift)) for _shift in range(len(_vals))]
        min_disorder_id = np.argmin(num_disorders)
        if debug:
            print('shift from: %d ' % min_disorder_id)
        # rearrange the points in a ring
        map_ids = RingShift(list(np.arange(len(_vals))), min_disorder_id)
        _angs = [_angs[_id] for _id in map_ids]
        _vals = [_vals[_id] for _id in map_ids]
        if debug:
            print('angles:')
            print(_angs)
            print('values:')
            print(_vals)
        ####### remap angles according to new angle origin ######
        origin = _angs[0]
        _angs = [_a - origin if _a >= origin else 2.*np.pi + _a - origin for _a in _angs]
        if debug:
            print('angles updated: ')
            print(_angs)
        # remove bad points acoording to the fact: values are monotonous to the angles
        # A greedy algorithm: each time remove the very point which bring the most disorder untill no disorder is found
        # check remained point volume, if insufficient, then this is a bad prediction
        while CountDisorder(_vals) > 0:
            num_disorders = [CountDisorder(_vals[:rm_id] + _vals[rm_id+1:]) for rm_id in range(len(_vals))]
            min_disorder_id = np.argmin(num_disorders)
            if debug:
                print(num_disorders)
                print('remove element at: %d ' % min_disorder_id)
            _angs = _angs[:min_disorder_id] + _angs[min_disorder_id+1:]
            _vals = _vals[:min_disorder_id] + _vals[min_disorder_id+1:]
        if debug:
            print('the final values: ')
            print(_vals)
        # check if points are sufficient for linear modeling
        if len(_vals) < 2:
            continue
        # get staged linear function
        # convert center coordinate from standard to original image
        center_pt = (centers[0][0] / im_scale - im_pad_w/2, centers[0][1] / im_scale - im_pad_h/2)
        linear_func = StagedLinearFunction(_angs, _vals, center_pt, rads[i], origin, debug=debug)
        # a demo call
        if debug:
            print(linear_func.stages)
            print(linear_func.params)
        ret_funcs += [linear_func]
    return ret_funcs

def TEST_GetDashboardReader(debug=False):
    test_double_ring = 'ds1.png'
    # test double ring dashboard
    im = cv2.imread(test_double_ring)
    ptr_box = (0,0,im.shape[1]/2,im.shape[0]/2)
    funcs = GetDashboardReader(im, debug=debug)
    if EXPECT_EQ(len(funcs), 2, 'DASHBOARD_T1'):
        # test outer ring
        EXPECT_EQ(funcs[0].num_stages, 0, 'DASHBOARD_T2')
        EXPECT_EQ_SET(funcs[0].stages, [], 'DASHBOARD_T3')
        EXPECT_EQ(abs(funcs[0](ptr_box)-0.5)<0.03, True, 'DASHBOARD_T4') # check residule
        print(funcs[0](ptr_box))
        # test inner ring
        EXPECT_EQ(funcs[1].num_stages, 0, 'DASHBOARD_T5')
        EXPECT_EQ_SET(funcs[1].stages, [], 'DASHBOARD_T6')
        EXPECT_EQ(abs(funcs[1](ptr_box)-8)<1, True, 'DASHBOARD_T7')
        print(funcs[1](ptr_box))

# for module test only, call this module seperatelly will run the test
if __name__ == '__main__':
    if _TEST_MODE_:
        # running test cases
        TEST_Kmeans(debug=_DEBUG_MODE_)
        TEST_GetDashboardReader(debug=_DEBUG_MODE_)
