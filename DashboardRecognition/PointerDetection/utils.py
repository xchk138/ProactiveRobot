# utils.py
import os
import cv2
import numpy as np

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

# convert from grey image into BGR color format, and enhance color
def Preprocess(im:np.ndarray) -> np.ndarray:
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
        return cv2.cvtColor(im, cv2.COLOR_YUV2BGR_I420)
    else:
        im = clahe.apply(im)
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)