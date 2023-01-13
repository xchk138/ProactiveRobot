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

# find which diagonal line is the pointer on dashboard
# return 0 stands for diagonal line from top left to bottom right
# return 1 stands for diagonal line from top right to bottom left
# return -1 stands for low confidence on judging.
def LocatePointer(im:np.ndarray, debug=False)->int:
    # split the rect of image into 4 parts of same size,
    # represents top-left,top-right,bottom-left,bottom-right regions
    if debug:
        cv2.imshow('locate pointer', im)
        cv2.waitKey(0)
    h, w = im.shape[:2]
    reg_top_left = im[:h//2, :w//2]
    reg_top_right = im[:h//2, w//2:]
    reg_bottom_left = im[h//2:, :w//2]
    reg_bottom_right = im[h//2:, w//2:]
    # if top left and bottom right region is literally darker,
    # then the diagonal line from TL to BR(back slash) is the pointer;
    # otherwise, the diagonal from TR to BL(slash) is the pointer.
    avg_TLBR = reg_top_left.mean() + reg_bottom_right.mean()
    avg_TRBL = reg_top_right.mean() + reg_bottom_left.mean()
    if abs(avg_TLBR - avg_TRBL) > 0.1*max(avg_TRBL, avg_TLBR):
        if avg_TLBR < avg_TRBL:
            if debug:
                print('TL to BR is darker, chosen for pointer')
            return 0
        else:
            if debug:
                print('TR to BL is darker, chosen for pointer')
            return 1
    else:
        if debug:
            print('TLBR and TRBL is equally bright, cannot chose!')
        return -1
