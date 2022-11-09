# make_coco.py
from matplotlib import image
import numpy as np
import cv2
import os
from enum import Enum

#np.random.seed(168)

LABEL_NAMES = ['YuanBiao', 'YeJing']


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
    return (np.round(r * 255), np.round(g * 255), np.round(b * 255))

def GenerateColorMap(size, chrom_thres=0.7, illum_range=(0.3, 0.6)):
    hue = np.random.rand(size)
    sat = np.random.rand(size) * (1.0-chrom_thres) + chrom_thres
    lig = np.random.rand(size) * (illum_range[1]-illum_range[0]) + illum_range[0]
    return [HSL2RGB(hue[i], sat[i], lig[i]) for i in range(size)]

LABEL_COLORS = GenerateColorMap(len(LABEL_NAMES), 0.7)
print(LABEL_COLORS)


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
        def __init__(self, bbox, label) -> None:
            self.bbox = bbox
            self.label = label

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
    def ShowDetection(im: np.ndarray, bboxes:list, labels:list):
        assert len(bboxes) == len(labels)
        for i in range(len(bboxes)):
            DrawConsole.draw_bbox(im, bboxes[i], labels[i])
            DrawConsole.add_label(im, bboxes[i], labels[i])
    @staticmethod
    def show_targets(im: np.ndarray, targets:list):
        for i in range(len(targets)):
            DrawConsole.draw_bbox(im, targets[i].bbox, targets[i].label)
            DrawConsole.add_label(im, targets[i].bbox, targets[i].label)

    def __init__(self, im, win_name='draw rect') -> None:
        self.state = DrawConsole.DrawState.END
        self.win_name = win_name
        self.im_ori = im
        self.im_tmp = im.copy()
        self.im_vis = im.copy()
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
            if np.any(VEC(self.stop_pos) - VEC(self.start_pos) == 0):
                print('no rect is drawn!')
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

    def update(self, pos):
        if self.state == DrawConsole.DrawState.BEGIN:
            pos = DrawConsole.restrict(pos, DrawConsole.shape_to_size(self.im_ori.shape))
            self.im_vis = self.im_tmp.copy()
            cv2.rectangle(self.im_vis, self.start_pos, pos, LABEL_COLORS[self.label], 2)
            cv2.imshow(self.win_name, self.im_vis)

    def cancel(self):
        if self.state == DrawConsole.DrawState.END:
            if len(self.targets) > 0:
                self.targets.pop()
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
            params.cancel()
    if event == cv2.EVENT_MOUSEMOVE:
        #print('mouse moved: %d %d' % (x, y))
        if params.state == DrawConsole.DrawState.BEGIN:
            params.update((x, y))
    if event == cv2.EVENT_RBUTTONUP:
        #print('right button clicked!')
        if params.state == DrawConsole.DrawState.END or params.state == DrawConsole.DrawState.MENU:
            params.showMenu((x, y))


def GetBoundingRectsAndLabels(im, default_label=0):
    vis = im.copy()
    # get different classes
    info = DrawConsole(vis)
    info.win_name = 'select color to draw rects'
    info.label = default_label
    cv2.imshow(info.win_name, vis)
    cv2.setMouseCallback(info.win_name, OnDrawRect, info)
    code = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if code == ord('c') or code == ord('C'):
        exit(0)
    return [info.targets[i].bbox for i in range(len(info.targets))], [info.targets[i].label for i in range(len(info.targets))]


def GetImages(image_path):
    filenames = os.listdir(image_path)
    i = 0
    for _fn in filenames:
        if _fn.endswith('.jpg') or _fn.endswith('.png'):
            _fn = os.path.join(image_path, _fn)
            im = cv2.imread(_fn, cv2.IMREAD_COLOR)
            if im is None:
                continue
            if len(im.shape)==2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            elif im.shape[-1] == 4:
                im = im[:,:,:3]
            yield (i, _fn, im)
            i += 1


def SaveImage(path_images, frame_id, im):
    cv2.imwrite('/'.join([path_images, '%08d.jpg' % frame_id]), im)


# works on single tracking
# coco format: [class_id] [x] [y] [w] [h]
def SaveLabels(path_labels, frame_id, bboxes, labels, w, h):
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
        label_file.write('%d %.7f %.7f %.7f %.7f\n' % (cid, bx, by, bw, bh))
    label_file.close()


def AppendSampleToList(file_list, images_dir, train_dir, frame_id):
    assert file_list.writable()
    file_list.write('/'.join(['.', images_dir, train_dir, '%08d.jpg\n' % frame_id]))
    file_list.flush()


if __name__ == '__main__':
    dataset_path = 'data/dashboard-coco'
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

    def SaveImageAndLabels(frame_id:int, im:np.ndarray, rects:list, labels:list) -> bool:
        if not len(rects):
            return False
        if np.random.rand() < train_ratio:
            SaveImage(path_images_train, frame_id, im)
            SaveLabels(path_labels_train, frame_id, rects, labels, im.shape[1], im.shape[0])
            AppendSampleToList(file_train_list, images_dir, train_dir, frame_id)
        else:
            SaveImage(path_images_val, frame_id, im)
            SaveLabels(path_labels_val, frame_id, rects, labels, im.shape[1], im.shape[0])
            AppendSampleToList(file_val_list, images_dir, val_dir, frame_id)
        return True

    images = GetImages('data/Baidu_YeJing')
    
    counter = 920
    begin_id = 0
    for i, fn, im in images:
        if i < begin_id:
            continue
        print("%d: %s" % (i, fn))
        rects, labels = GetBoundingRectsAndLabels(im, default_label=1)
        print(rects)
        print(labels)
        #print(im.shape[::-1])
        # save the frame with bounding box
        if SaveImageAndLabels(counter, im, rects, labels):
            counter += 1
            print('counter=%d' % counter)