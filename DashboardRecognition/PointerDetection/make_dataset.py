# make_dataset.py

# 1. Load images
# 2. Load Dashboard Detection Model
# 3. Detect Dashboards in single image
# 4. Draw Pointer on each image
# 5. Crop every dashboard out and save them
# 6. Save the pointer information related to the dashboard image.

from matplotlib import image
import numpy as np
import cv2
import os
from enum import Enum
import platform

from inference import YoloDetector
from utils import GetImages, Preprocess

# USE KEYPOINT
USE_KEYPOINT = False
NUM_KEYPOINTS = 2
BBOX_TYPES = ['Dashboard', 'Display', 'Pointer']
if USE_KEYPOINT:
    LABEL_NAMES = [*BBOX_TYPES, 'Keypoint']
else:
    LABEL_NAMES = BBOX_TYPES


def hue2rgb(p, q, t):
            if t < 0: t += 1
            if t > 1: t -= 1
            if t < 1/6.0: return p + (q - p) * 6 * t
            if t < 1/2.0: return q
            if t < 2/3.0: return p + (q - p) * (2/3.0 - t) * 6
            return p

def HSL2RGB(h, s, l):
    if s == 0:
        r = g = b = l
    else:
        if l < 0.5:
            q = l * (1 + s)
        else:
            q = l + s - l * s
        p = 2 * l - q
        r = hue2rgb(p, q, h + 1/3.0)
        g = hue2rgb(p, q, h)
        b = hue2rgb(p, q, h - 1/3.0)
    return (int(np.round(r * 255)), int(np.round(g * 255)), int(np.round(b * 255)))

def GenerateColorMap(size, chrom_thres=0.8, illum_range=(0.2, 0.5)):
    hue = np.random.rand(size)
    sat = np.random.rand(size) * (1.0-chrom_thres) + chrom_thres
    lig = np.random.rand(size) * (illum_range[1]-illum_range[0]) + illum_range[0]
    return [HSL2RGB(hue[i], sat[i], lig[i]) for i in range(size)]

#LABEL_COLORS = GenerateColorMap(len(LABEL_NAMES), 0.8)
LABEL_COLORS = [(200,30,0), (0, 200, 30), (30, 0, 200), (180, 100, 0)]


def VEC(x):
    return np.array(x)

def TUPLE(x: np.ndarray):
    return tuple(x.tolist())


# struct for draw Console
class DrawConsole(object):
    class DrawState(Enum):
        BEGIN = 1
        END = 2
        MENU = 3
    
    class Target(object):
        def __init__(self, bbox=(0,0,0,0), label=0, kpts=[]) -> None:
            self.bbox = bbox
            self.label = label
            self.kpts = kpts

    @staticmethod
    def restrict(pos, size):
        x = min(size[0], max(pos[0], 0))
        y = min(size[1], max(pos[1], 0))
        return (x, y)
    @staticmethod
    def shape_to_size(shape):
        return (shape[1], shape[0])
    @staticmethod
    def draw_bbox(im, bbox:tuple, label:int) -> None:
        cv2.rectangle(im, tuple(bbox[0:2]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), LABEL_COLORS[label], 2)
    @staticmethod
    def draw_keypoints(im, kpts:list) -> None:
        for i in range(len(kpts)):
            cv2.circle(im, kpts[i], 6, LABEL_COLORS[-1], -1)
        if len(kpts) >= 2:
            cv2.line(im, kpts[0], kpts[1], LABEL_COLORS[-1], 3)
    @staticmethod
    def add_label(im:np.ndarray, bbox:tuple, label:int)->None:
        font_size = 0.5
        font_weight = 2
        font_color = (255, 255, 255)
        font_face = cv2.FONT_HERSHEY_COMPLEX
        label_size = cv2.getTextSize(LABEL_NAMES[label], font_face, font_size, font_weight)[0]
        padding = 3
        rect_size = (label_size[0], label_size[1] + 2*padding)
        rect_begin_pos = (bbox[0], bbox[1] - rect_size[1])
        text_begin_pos = (bbox[0], bbox[1] - padding)
        if rect_begin_pos[1] < 0:
            rect_begin_pos = (bbox[0], bbox[1])
            text_begin_pos = (bbox[0], bbox[1] + rect_size[1] - padding)
        rect_end_pos = TUPLE(VEC(rect_begin_pos) + VEC(rect_size))
        cv2.rectangle(im, rect_begin_pos, rect_end_pos, LABEL_COLORS[label], -1)
        cv2.putText(im, LABEL_NAMES[label], text_begin_pos, font_face, font_size, font_color, font_weight)
    @staticmethod
    def menu_rect(menu_pos, cell_size):
        return (*menu_pos, menu_pos[0] + cell_size[0], menu_pos[1] + cell_size[1]*len(LABEL_NAMES))
    @staticmethod
    def point_in_rect(pos:tuple, rect:tuple):
        return (pos[0] > rect[0] and pos[0] < rect[2] and pos[1] > rect[1] and pos[1] < rect[3])
    @staticmethod
    def ShowDetection(im: np.ndarray, bboxes:list, labels:list, kpts=[]):
        assert len(bboxes) == len(labels)
        for i in range(len(bboxes)):
            DrawConsole.draw_bbox(im, bboxes[i], labels[i])
            DrawConsole.draw_keypoints(im, kpts[i])
            DrawConsole.add_label(im, bboxes[i], labels[i])
    @staticmethod
    def show_targets(im: np.ndarray, targets:list):
        for i in range(len(targets)):
            DrawConsole.draw_bbox(im, targets[i].bbox, targets[i].label)
            DrawConsole.draw_keypoints(im, targets[i].kpts)
            DrawConsole.add_label(im, targets[i].bbox, targets[i].label)
    @staticmethod
    def point_in_bbox(pos:tuple, bbox:tuple):
        rect = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
        return DrawConsole.point_in_rect(pos, rect)
    @staticmethod
    def make_alike(x: np.ndarray):
        return np.zeros_like(x, x.dtype)
    @staticmethod
    def float_alike(x: np.ndarray):
        return np.zeros_like(x, np.float32)
    @staticmethod
    def calc_area(x, y):
        return abs(x[0] - y[0]) * abs(x[1] - y[1])
    @staticmethod
    def is_rect_illegal(x, y):
        return abs(x[0] - y[0]) > 10 and abs(x[1] - y[1]) > 10

    def __init__(self, im, win_name='draw rect') -> None:
        self.state = DrawConsole.DrawState.END
        self.win_name = win_name
        self.im_ori = im
        self.im_tmp = im.copy()
        self.im_vis = im.copy()
        self.im_mask = None
        self.im_alpha = None
        self.start_pos = (0, 0)
        self.stop_pos = (0, 0)
        self.label = 0
        self.bbox = (0, 0, 0, 0)
        self.menu_pos = (0, 0)
        self.menu_cell_size = (100, 20)
        self.targets = [] # Target type

    def start(self, pos):
        if self.state == DrawConsole.DrawState.END:
            pos = DrawConsole.restrict(pos, DrawConsole.shape_to_size(self.im_ori.shape))
            self.start_pos = pos
            self.state = DrawConsole.DrawState.BEGIN

    def stop(self, pos):
        if self.state == DrawConsole.DrawState.BEGIN:
            self.state = DrawConsole.DrawState.END
            pos = DrawConsole.restrict(pos, DrawConsole.shape_to_size(self.im_ori.shape))
            self.stop_pos = pos
            if LABEL_NAMES[self.label] in BBOX_TYPES:
                # add bbox to targets
                if not DrawConsole.is_rect_illegal(self.start_pos, self.stop_pos):
                    #print('no rect is drawn!')
                    self.bbox = (0, 0, 0, 0)
                    self.im_vis = self.im_tmp.copy()
                    cv2.imshow(self.win_name, self.im_vis)
                else:
                    self.bbox = (min(self.start_pos[0], self.stop_pos[0]),
                        min(self.start_pos[1], self.stop_pos[1]), 
                        abs(self.start_pos[0] - self.stop_pos[0]), 
                        abs(self.start_pos[1] - self.stop_pos[1]))
                    # draw labels on the top left of the rect
                    DrawConsole.add_label(self.im_vis, self.bbox, self.label)
                    cv2.imshow(self.win_name, self.im_vis)
                    # add this target to list
                    self.targets.append(DrawConsole.Target(self.bbox, self.label))
                    self.im_tmp = self.im_vis.copy()
            elif LABEL_NAMES[self.label] in ['Keypoint']:
                # add keypoints to nearest bbox
                if max(abs(self.stop_pos[0] - self.start_pos[0]), abs(self.stop_pos[1] - self.start_pos[1])) < 10:
                    #print('no keypoint is drawn')
                    self.im_vis = self.im_tmp.copy()
                    cv2.imshow(self.win_name, self.im_vis)
                else:
                    #print('keypoint: (%d,%d)-(%d,%d)' % (self.start_pos[0], self.start_pos[1], self.stop_pos[0], self.stop_pos[1]))
                    # find if all these points are inside a box
                    kpts_valid = False
                    kpts = [self.start_pos, self.stop_pos]
                    for i in range(len(self.targets)):
                        is_inside = [DrawConsole.point_in_bbox(kpts[j], self.targets[i].bbox) for j in range(len(kpts))]
                        if np.all(is_inside):
                            # bind these key points to this bounding box
                            self.targets[i].kpts = kpts
                            kpts_valid = True
                            break
                    if not kpts_valid:
                        #print('keypoints are not all inside a boundg box!')
                        self.im_vis = self.im_tmp.copy()
                        cv2.imshow(self.win_name, self.im_vis)
                    else:
                        # redraw all rects and lines
                        self.im_vis = self.im_ori.copy()
                        DrawConsole.show_targets(self.im_vis, self.targets)
                        self.im_tmp = self.im_vis.copy()
                        cv2.imshow(self.win_name, self.im_vis)
                
    def update(self, pos):
        if self.state == DrawConsole.DrawState.BEGIN:
            pos = DrawConsole.restrict(pos, DrawConsole.shape_to_size(self.im_ori.shape))
            self.im_vis = self.im_tmp.copy()
            if LABEL_NAMES[self.label] in BBOX_TYPES:
                cv2.rectangle(self.im_vis, self.start_pos, pos, LABEL_COLORS[self.label], 2)
            elif LABEL_NAMES[self.label] in ['Keypoint']:
                cv2.line(self.im_vis, self.start_pos, pos, LABEL_COLORS[self.label], 3)
            cv2.imshow(self.win_name, self.im_vis)
        elif self.state == DrawConsole.DrawState.END:
            self.im_vis = self.im_tmp.copy()
            self.im_alpha = DrawConsole.make_alike(self.im_vis)
            self._im_w = DrawConsole.make_alike(self.im_vis)
            for i in range(len(self.targets)):
                bb = self.targets[i].bbox
                if DrawConsole.point_in_bbox(pos, bb):
                    rect = (bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3])
                    _color = LABEL_COLORS[self.targets[i].label]
                    cv2.rectangle(self.im_alpha, rect[:2], rect[2:], _color, -1)
                    cv2.rectangle(self._im_w, rect[:2], rect[2:], (255,255,255), -1)
            # merge alpha layer and rgb layer
            self.im_mask = np.float32(self._im_w) * (0.5/255.0)
            self._alpha = np.float32(self.im_alpha) * self.im_mask
            self._vis = np.float32(self.im_vis) * (1.0 - self.im_mask)
            self.im_vis = np.uint8(self._alpha + self._vis)
            cv2.imshow(self.win_name, self.im_vis)

    def cancel(self, pos):
        if self.state == DrawConsole.DrawState.END:
            if len(self.targets) > 0:
                # find the bounding boxes that has been clicked
                new_targets = []
                is_clicked = []
                for i in range(len(self.targets)):
                    # if pos is in rect
                    if not DrawConsole.point_in_bbox(pos, self.targets[i].bbox):
                        new_targets.append(DrawConsole.Target(self.targets[i].bbox, self.targets[i].label, self.targets[i].kpts))
                    else:
                        is_clicked.append(i)
                if len(is_clicked) == 1:
                    self.label = self.targets[is_clicked[0]].label
                elif len(is_clicked) > 1:
                    # find the smallest one 
                    _min_area = 1e6
                    _min_id = -1
                    for i in range(len(is_clicked)):
                        bb = self.targets[is_clicked[i]].bbox
                        _area = bb[2]*bb[3]
                        if _min_area >= _area:
                            _min_id = i
                            _min_area = _area
                    # remove the smallest one
                    for i in range(len(is_clicked)):
                        if i != _min_id:
                            new_targets.append(self.targets[is_clicked[i]])
                    # set the current label as the deleted one
                    self.label = self.targets[is_clicked[_min_id]].label
                self.targets = new_targets
                self.im_tmp = self.im_ori.copy()
                DrawConsole.show_targets(self.im_tmp, self.targets)
                self.im_vis = self.im_tmp.copy()
                cv2.imshow(self.win_name, self.im_vis)

    def showMenu(self, pos):
        if self.state == DrawConsole.DrawState.END or self.state == DrawConsole.DrawState.MENU:
            self.state = DrawConsole.DrawState.MENU
            self.im_vis = self.im_tmp.copy()
            label_size = self.menu_cell_size
            self.menu_pos = pos
            for i in range(len(LABEL_NAMES)):
                cell_begin_pos = (pos[0], pos[1] + i * label_size[1])
                cell_end_pos = (pos[0] + label_size[0], pos[1] + (i+1) * label_size[1])
                cv2.rectangle(self.im_vis, cell_begin_pos, cell_end_pos, LABEL_COLORS[i], -1)
                DrawConsole.add_label(self.im_vis, (pos[0], pos[1] + (i+1)*label_size[1], 0,0), i)
            cv2.imshow(self.win_name, self.im_vis)

    def closeMenu(self, pos):
        if self.state == DrawConsole.DrawState.MENU:
            if not DrawConsole.point_in_rect(pos, DrawConsole.menu_rect(self.menu_pos, self.menu_cell_size)):
                self.im_vis = self.im_tmp.copy()
                cv2.imshow(self.win_name, self.im_vis)
                self.state = DrawConsole.DrawState.END
    
    def selectLabel(self, pos):
        if self.state == DrawConsole.DrawState.MENU:
            if DrawConsole.point_in_rect(pos, DrawConsole.menu_rect(self.menu_pos, self.menu_cell_size)):
                # find which menu cell is on
                self.label = (pos[1] - self.menu_pos[1]) // self.menu_cell_size[1]
                self.im_vis = self.im_tmp.copy()
                cv2.rectangle(self.im_vis, (0, 0), self.menu_cell_size, LABEL_COLORS[ self.label], -1)
                DrawConsole.add_label(self.im_vis, (0, self.menu_cell_size[1], 0,0),  self.label)
                cv2.imshow(self.win_name, self.im_vis)
                self.state = DrawConsole.DrawState.END


def OnDrawRect(event, x, y, flags, params:DrawConsole):
    if  event == cv2.EVENT_LBUTTONDOWN:
        #print("mouse click down: %d %d"%(x,y))
        if params.state == DrawConsole.DrawState.END:
            params.start((x, y))
        elif params.state == DrawConsole.DrawState.MENU:
            params.closeMenu((x, y))
    if  event == cv2.EVENT_LBUTTONUP:
        #print("mouse click up: %d %d" % (x, y))
        if params.state == DrawConsole.DrawState.BEGIN:
            params.stop((x, y))
        elif params.state == DrawConsole.DrawState.MENU:
            params.selectLabel((x, y))
    if  event == cv2.EVENT_LBUTTONDBLCLK:
        #print('mouse double clicked, task cancelled.')
        if params.state == DrawConsole.DrawState.END:
            params.cancel((x,y))
    if event == cv2.EVENT_MOUSEMOVE:
        #print('mouse moved: %d %d' % (x, y))
        if params.state in [DrawConsole.DrawState.BEGIN, DrawConsole.DrawState.END]:
            params.update((x, y))
    if event == cv2.EVENT_RBUTTONDOWN:
        if params.state == DrawConsole.DrawState.END or params.state == DrawConsole.DrawState.MENU:
            params.showMenu((x, y))


def GetBoundingRectsAndLabels(im, default_label=0, backend=None):
    im = Preprocess(im)
    vis = im.copy()
    info = DrawConsole(vis)
    info.win_name = 'draw rects and pointers'
    info.label = default_label

    # use pretrained model to generate coarse detection result
    if backend is not None:
        #bboxes, labels = backend.infer_detail(im)
        #bboxes, labels = backend.infer(im)
        bboxes, labels, _ = backend.infer_closer(im)
        for i in range(len(bboxes)):
            if LABEL_NAMES[labels[i]] in BBOX_TYPES:
                info.targets.append(DrawConsole.Target(bboxes[i], labels[i]))
                DrawConsole.show_targets(info.im_tmp, info.targets)
    if platform.system() == 'Windows':
        cv2.namedWindow(info.win_name)
        #cv2.setWindowProperty(info.win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow(info.win_name, cv2.WINDOW_GUI_NORMAL)
        cv2.setWindowProperty(info.win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(info.win_name, info.im_tmp)
    cv2.setMouseCallback(info.win_name, OnDrawRect, info)
    code = -1
    while code not in [ord('c'), ord('C'), ord(' ')]:
        code = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if code == ord('c') or code == ord('C'):
        exit(0)
    return ([info.targets[i].bbox for i in range(len(info.targets))], 
    [info.targets[i].label for i in range(len(info.targets))],
    [info.targets[i].kpts for i in range(len(info.targets))])

def SaveImage(path_images, frame_id, im):
    cv2.imwrite('/'.join([path_images, '%08d.jpg' % frame_id]), im)


# works on single tracking
# coco format: [class_id] [x] [y] [w] [h]
def SaveLabels(path_labels, frame_id, bboxes, labels, kpts, w, h):
    assert len(bboxes) == len(labels)
    label_file = open('/'.join([path_labels, '%08d.txt' % frame_id]), 'wt')
    for i in range(len(bboxes)):
        cid = labels[i]
        bbox = bboxes[i]
        # create a txt file
        bx = bbox[0] * 1.0 / w
        by = bbox[1] * 1.0 / h
        bw = bbox[2] * 1.0 / w
        bh = bbox[3] * 1.0 / h
        # restrict to range
        bx = min(1.0, max(0.0, bx))
        by = min(1.0, max(0.0, by))
        bw = min(1.0, max(0.0, bw))
        bh = min(1.0, max(0.0, bh))
        bx += bw/2
        by += bh/2
        if USE_KEYPOINT:
            # normalize keypoint coordinates
            _kps = np.zeros([NUM_KEYPOINTS, 3], np.float32)
            if len(kpts[i]) == NUM_KEYPOINTS:
                for j in range(NUM_KEYPOINTS):
                    _kps[j,0] = kpts[i][j][0] * 1.0 / w
                    _kps[j,1] = kpts[i][j][1] * 1.0 / h
                    _kps[j,2] = 1.0
            _kps = _kps.reshape([NUM_KEYPOINTS*3])
            _kps = ['%.7f'%_e for _e in _kps.tolist()]
            _kps_str = ' ' + (' '.join(_kps))
        else:
            _kps_str = ''
        label_file.write('%d %.7f %.7f %.7f %.7f%s\n' % (cid, bx, by, bw, bh, _kps_str))
    label_file.close()


def AppendSampleToList(file_list, images_dir, train_dir, frame_id):
    assert file_list.writable()
    file_list.write('/'.join(['.', images_dir, train_dir, '%08d.jpg\n' % frame_id]))
    file_list.flush()


if __name__ == '__main__':
    dataset_path = '../data/pointer'
    images_dir = 'images'
    labels_dir = 'labels'
    train_dir = 'train'
    val_dir = 'val'
    train_ratio = 0.8
    
    # create dirs 
    path_images_train = '/'.join([dataset_path, images_dir, train_dir])
    os.makedirs(path_images_train, exist_ok=True)
    path_images_val = '/'.join([dataset_path, images_dir, val_dir])
    os.makedirs(path_images_val, exist_ok=True)
    path_labels_train = '/'.join([dataset_path, labels_dir, train_dir])
    os.makedirs(path_labels_train, exist_ok=True)
    path_labels_val = '/'.join([dataset_path, labels_dir, val_dir])
    os.makedirs(path_labels_val, exist_ok=True)
    # generate text files 
    path_train_list = '/'.join([dataset_path, train_dir]) + '.txt'
    path_val_list = '/'.join([dataset_path, val_dir]) + '.txt'
    file_train_list = open(path_train_list, 'at')
    file_val_list = open(path_val_list, 'at')

    def SaveImageAndLabels(frame_id:int, im:np.ndarray, rects:list, labels:list, kpts:list) -> bool:
        if not len(rects):
            return False
        if np.random.rand() < train_ratio:
            SaveImage(path_images_train, frame_id, im)
            SaveLabels(path_labels_train, frame_id, rects, labels, kpts, im.shape[1], im.shape[0])
            AppendSampleToList(file_train_list, images_dir, train_dir, frame_id)
        else:
            SaveImage(path_images_val, frame_id, im)
            SaveLabels(path_labels_val, frame_id, rects, labels, kpts, im.shape[1], im.shape[0])
            AppendSampleToList(file_val_list, images_dir, val_dir, frame_id)
        return True
    
    images = GetImages('../data/samples/real/2')
    #images = GetImages('../data/raw/2_YaLiBiao')
    detector = YoloDetector('../models/yolov5n-pointer-1.onnx', num_class=3, score_thres=0.3, conf_thres=0.3)

    counter = 10023
    #begin_id = 1252
    begin_id = 0
    for i, fn, im in images:
        if i < begin_id:
            continue
        print("GLOBAL(%d)-LOCAL(%d): %s" % (counter, i, fn))
        rects, labels, kpts = GetBoundingRectsAndLabels(im, default_label=2, backend=detector)
        #if SaveImageAndLabels(counter, im, rects, labels, kpts):
        #  counter += len(rects)