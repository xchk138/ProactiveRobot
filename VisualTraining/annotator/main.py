import os
import cv2
import numpy as np


class Eye(object):
    def __init__(self) -> None:
        params = cv2.TrackerDaSiamRPN_Params()
        params.kernel_cls1 = 'annotator/SiamRPN/dasiamrpn_kernel_cls1.onnx'
        params.kernel_r1 = 'annotator/SiamRPN/dasiamrpn_kernel_r1.onnx'
        params.model = 'annotator/SiamRPN/dasiamrpn_model.onnx'
        self.tracker = cv2.TrackerDaSiamRPN_create(params)
    def lookAt(self, im, bbox):
        self.bbox = bbox
        self.tracker.init(im, bbox)
    def track(self, im):
        _, self.bbox = self.tracker.update(im)
        #print(self.bbox)
        return self.tracker.getTrackingScore()



def VisualizeBoundingBox(_im, rects):
    for _rect in rects:
        _im = cv2.rectangle(
            _im, 
            (_rect[0], _rect[1]), 
            (_rect[0] + _rect[2], _rect[1] + _rect[3]), 
            (0, 100, 255), 
            2, 8)
    return _im


# center crop
def CropImage(im_raw):
    h, w, c = im_raw.shape
    assert c==3
    if w > h:
        crop_width = (w - h) // 2
        return im_raw[0:h, crop_width:crop_width+h]
    else:
        crop_height = (h - w) // 2
        return im_raw[crop_height:crop_height+w, 0:w]


# resize image 
def ProcessImage(im_raw, target_size):
    return cv2.resize(im_raw, target_size)


def SetTarget(eye: Eye, im: np.ndarray):
    rect = cv2.selectROI('draw target rect', im)
    print(rect)
    if np.all(np.array(rect)==0):
        return False
    else:
        eye.lookAt(im, rect) # setup target
        return True


def SkipFrames(vid, num_frames):
    iFrame = 0
    _im = None
    while iFrame < num_frames:
        status, _im = vid.read()
        if not status:
            print('video end')
            exit(1)
        iFrame += 1
    return _im


def SaveImage(path_images, frame_id, im):
    cv2.imwrite('/'.join([path_images, '%08d.jpg' % frame_id]), im)


# works on single tracking
# coco format: [class_id] [x] [y] [w] [h]
def SaveLabel(path_labels, frame_id, bbox, w, h):
    cid = 0 # single class
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
    label_file = open('/'.join([path_labels, '%08d.txt' % frame_id]), 'wt')
    label_file.write('%d %.7f %.7f %.7f %.7f\n' % (cid, bx, by, bw, bh))
    label_file.close()


def AppendSampleToList(file_list, images_dir, train_dir, frame_id):
    assert file_list.writable()
    file_list.write('/'.join(['.', images_dir, train_dir, '%08d.jpg\n' % frame_id]))
    file_list.flush()


if __name__ == '__main__':
    # configurations
    target_size = (640, 360)
    min_score = 0.85
    skip_frames = 60
    dataset_path = 'data/dock-coco'
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
    file_train_list = open(path_train_list, 'wt')
    file_val_list = open(path_val_list, 'wt')

    def SaveImageAndLabel(frame_id, im):
        if np.random.rand() < train_ratio:
            SaveImage(path_images_train, frame_id, im)
            SaveLabel(path_labels_train, frame_id, eye.bbox, im.shape[1], im.shape[0])
            AppendSampleToList(file_train_list, images_dir, train_dir, frame_id)
        else:
            SaveImage(path_images_val, frame_id, im)
            SaveLabel(path_labels_val, frame_id, eye.bbox, im.shape[1], im.shape[0])
            AppendSampleToList(file_val_list, images_dir, val_dir, frame_id)

    frame_id = 0
    vid = cv2.VideoCapture('data/dock.mp4')
    eye = Eye()
    status, im_raw = vid.read()
    if not status:
        raise NameError("failed to read video file or stream")
    
    im = ProcessImage(im_raw, target_size)
    # setup the tracking target
    while not SetTarget(eye, im):
        im = ProcessImage(SkipFrames(vid, skip_frames), target_size)
    # save the first frame with bounding box
    SaveImageAndLabel(frame_id, im)
    frame_id += 1
    
    while True:
        status, im_raw = vid.read()
        if not status:
            break
        im = ProcessImage(im_raw, target_size)
        score = eye.track(im)
        if score < min_score:
            while not SetTarget(eye, im):
                im = ProcessImage(SkipFrames(vid, skip_frames), target_size)
        # save the frame with bounding box
        SaveImageAndLabel(frame_id, im)
        frame_id += 1
        im = VisualizeBoundingBox(im, [eye.bbox])
        cv2.imshow('dock', im)
        cv2.waitKey(30)

    # finalization
    file_train_list.close()
    file_val_list.close()
    print('Done.')
