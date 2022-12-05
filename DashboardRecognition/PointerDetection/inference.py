# inference.py

import cv2
import numpy as np

class YoloDetector(object):
    def __init__(self, onnx_path, infer_size=224, num_class=2, score_thres=0.6, conf_thres=0.6) -> None:
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.infer_size = infer_size
        self.num_class = num_class
        self.score_threshold = score_thres
        self.nms_threshold = 0.45
        self.confidence_threshold = conf_thres
        self.iou_threshold = 0.6
        # custom data
        self.pad_size = (0, 0)
        self.input_size = (0, 0)
        self.size_padded = 0
    
    def square_pad(self, image:np.ndarray):
        h, w, _ = image.shape
        self.input_size = (w, h)
        dim_diff = np.abs(h - w)
        pad1, pad2= dim_diff//2, dim_diff-dim_diff//2
        if h < w:
            image = cv2.copyMakeBorder(image,pad1,pad2,0,0,cv2.BORDER_CONSTANT,value=0)
            self.pad_size = (0, pad1)
            self.size_padded = w
        else:
            image = cv2.copyMakeBorder(image,0,0,pad1,pad2,cv2.BORDER_CONSTANT,value=0)
            self.pad_size = (pad1, 0)
            self.size_padded = h
        return image
    
    @staticmethod
    def format_color(image):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise NameError("image format error! unexpected shape: %s" % image.shape)
        return image
    
    def preprocess(self, image:np.ndarray):
        image = YoloDetector.format_color(image)
        image = self.square_pad(image)
        blob = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (self.infer_size, self.infer_size), None, True, False)
        return blob

    def postprocess(self, output):
        confidences = []
        boxes = []
        classes = []
        data = output[0]

        for i in range(data.shape[0]):
            confidence = data[i,4]
            # Discard bad detections and continue.
            if confidence >= self.confidence_threshold:
                scores = data[i,5:5+self.num_class]
                # acquire the index of best class score.
                class_id = np.argmax(scores)
                max_class_score = scores[class_id]
                # continue if the class score is above the threshold.
                if max_class_score > self.score_threshold:
                    # Store class ID and confidence in the pre-defined respective vectors.
                    confidences.append(confidence * max_class_score)
                    classes.append(class_id)
                    cx = data[i,0]
                    cy = data[i,1]
                    w = data[i,2]
                    h = data[i,3]
                    # Bounding box coordinates.
                    left = int(cx - 0.5 * w)
                    top = int(cy - 0.5 * h)
                    width = int(w)
                    height = int(h)
                    # Store good detections in the boxes vector.
                    boxes.append((left, top, width, height))
        
        # Perform Non-Maximum Suppression and draw predictions.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.score_threshold, self.nms_threshold)

        bboxes_ret = []
        labels_ret = []
        for i in range(len(indices)):
            bboxes_ret.append(boxes[indices[i]])
            labels_ret.append(classes[indices[i]])
        return self.remap_rects(bboxes_ret, labels_ret)

    def restrict(self, bboxes, labels, min_area=20):
        bboxes_ret = []
        labels_ret = []
        w, h = self.input_size
        for i in range(len(bboxes)):
            bx, by, bw, bh = bboxes[i]
            x1 = min(max(bx, 0), w-1)
            y1 = min(max(by, 0), h-1)
            x2 = min(max(bx+bw, 0), w-1)
            y2 = min(max(by+bh, 0), h-1)
            bx, by = int(x1), int(y1)
            bw = int(x2 - x1)
            bh = int(y2 - y1)
            if bw>0 and bh>0 and bw*bh > min_area:
                bboxes_ret.append((bx,by,bw,bh))
                labels_ret.append(labels[i])
        return bboxes_ret, labels_ret

    def remap_rects(self, bboxes, labels):
        scale_ = 1.0*self.size_padded / self.infer_size
        px = self.pad_size[0]
        py = self.pad_size[1]
        bboxes = [(x*scale_-px, y*scale_-py, w*scale_, h*scale_) for x,y,w,h in bboxes]
        return self.restrict(bboxes, labels)

    def infer(self, im:np.ndarray):
        bboxes = []
        labels = []
        x = self.preprocess(im)
        self.net.setInput(x)
        y = self.net.forward('output')
        bboxes, labels = self.postprocess(y)
        return bboxes, labels

    class ImagePatch(object):
        def __init__(self, im:np.ndarray, scale:float, offset:tuple) -> None:
            self.im = im
            self.scale = scale
            self.offset = offset
        @property
        def get_range(self):
            h, w = self.im.shape[:2]
            h_full = int(h/self.scale)
            w_full = int(w/self.scale)
            x_full = int(self.offset[0]/self.scale)
            y_full = int(self.offset[1]/self.scale)
            return (x_full, y_full, w_full, h_full)
        def map_rect(self, rect):
            x,y,w,h = self.get_range
            #print(x,y,w,h)
            rx = int(rect[0]/self.scale)
            ry = int(rect[1]/self.scale)
            rw = min(w-1-rx, int(rect[2]/self.scale))
            rh = min(h-1-ry, int(rect[3]/self.scale))
            rx += x
            ry += y
            return (rx, ry, rw, rh)
        def __str__(self) -> str:
            return 'patch {\n\tshape: (%d, %d)\n\tscale: %.6f\n\toffset: (%d, %d)\n\t}' % (self.im.shape[1], self.im.shape[0], self.scale, self.offset[0], self.offset[1])

    def split_image(self, im:np.ndarray, min_object_size=120, max_depth=3):
        patch_size = self.infer_size
        overlap = min_object_size
        h, w = im.shape[:2]
        scale = patch_size*1.0/min(w, h) # the basic scale
        ratio = 2
        patches = []
        depth = 1
        while scale <= 1 and depth <= max_depth:
            #print('depth: %d, scale: %f' % (depth, scale))
            # do splits
            _w = int(w * scale)
            _h = int(h * scale)
            _im = cv2.resize(im, (_w, _h))
            x_split = (_w - overlap)//(patch_size-overlap)
            y_split = (_h - overlap)//(patch_size-overlap)
            for _iy in range(y_split):
                patch_y = int(_iy*(patch_size-overlap))
                if _iy == y_split-1:
                    patch_h = int(_h - 1 - patch_y)
                else:
                    patch_h = patch_size
                for _ix in range(x_split):
                    patch_x = int(_ix*(patch_size-overlap))
                    if _ix == x_split-1:
                        patch_w = int(_w - 1 - patch_x)
                    else:
                        patch_w = patch_size
                patches.append(YoloDetector.ImagePatch(_im[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w], scale, (patch_x, patch_y)))
            # update scale and depth
            scale *= ratio
            depth += 1
        return patches

    @staticmethod
    def inter(bb1, bb2):
        bx1 = max(bb1[0], bb2[0])
        by1 = max(bb1[1], bb2[1])
        bx2 = min(bb1[0] + bb1[2], bb2[0] + bb2[2])
        by2 = min(bb1[1] + bb1[3], bb2[1] + bb2[3])
        if bx2 > bx1 and by2 > by1:
            return (bx1, by1, bx2-bx1, by2-by1)
        else:
            return None
    @staticmethod
    def union(bb1, bb2):
        bx1 = min(bb1[0], bb2[0])
        by1 = min(bb1[1], bb2[1])
        bx2 = max(bb1[0] + bb1[2], bb2[0] + bb2[2])
        by2 = max(bb1[1] + bb1[3], bb2[1] + bb2[3])
        return (bx1, by1, bx2-bx1, by2-by1)
    @staticmethod
    def area(bb):
        return bb[2]*bb[3]
    @staticmethod
    def smaller(bb1, bb2):
        if YoloDetector.area(bb1) > YoloDetector.area(bb2):
            return bb2
        else:
            return bb1
    @staticmethod
    def bigger(bb1, bb2):
        if YoloDetector.area(bb1) > YoloDetector.area(bb2):
            return bb1
        else:
            return bb2
    @staticmethod
    def iou(bb1, bb2):
        _overlap = YoloDetector.inter(bb1, bb2)
        if _overlap is None:
            return 0
        else:
            return YoloDetector.area(_overlap) *1.0 / YoloDetector.area(YoloDetector.smaller(bb1, bb2))

    @staticmethod
    def merge_bboxes(bboxes, labels, max_iou=0.6):
        is_removed = [False]*len(bboxes)
        for i in range(len(bboxes)):
            if is_removed[i]:
                continue
            for j in range(i+1, len(bboxes)):
                if is_removed[j]:
                    continue
                if labels[i] != labels[j]:
                    continue
                if YoloDetector.iou(bboxes[i], bboxes[j]) >= max_iou:
                    # mark the small one as removed
                    if YoloDetector.area(bboxes[i]) > YoloDetector.area(bboxes[j]):
                        is_removed[j] = True
                    else:
                        is_removed[i] = True
        bbs = []
        lbs = []
        for i in range(len(is_removed)):
            if not is_removed[i]:
                bbs.append(bboxes[i])
                lbs.append(labels[i])
        return bbs, lbs


    def infer_detail(self, im:np.ndarray):
        bboxes = []
        labels = []
        x_batch = self.split_image(im)
        #[print(x_batch[i]) for i in range(len(x_batch))]
        for x in x_batch:
            _bboxes, _labels = self.infer(x.im)
            bboxes += [x.map_rect(bb) for bb in _bboxes]
            labels += _labels
        return YoloDetector.merge_bboxes(bboxes, labels)
