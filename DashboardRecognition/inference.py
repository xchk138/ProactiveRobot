# inference.py

import cv2
import numpy as np

class YoloDetector(object):
    def __init__(self, onnx_path, infer_size=224, num_class=2, score_thres=0.8, conf_thres=0.8) -> None:
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
