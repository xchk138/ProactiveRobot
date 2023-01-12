from extract_digits import GetDashboardReader
import cv2

if __name__ == '__main__':
    im = cv2.imread('t2.png')
    ptr_box = (0,0,im.shape[1]/2,im.shape[0])
    funcs = GetDashboardReader(im, debug=True)
    print(len(funcs))
    for func in funcs:
        print(func.num_stages)
        print(func.stages)
        print(func.params)
        print(func(ptr_box))
        