

import numpy as np 
import cv2

def get_filename(path):
    return path.split('/')[-1]

def change_suffix(s, new_suffix, index=None):
    i = s.rindex('.')
    si = ""
    if index is not None:
        si = "_" + str(index)
    s = s[:i] + si + "." + new_suffix
    return s 


def int2str(num, len):
    return ("{:0"+str(len)+"d}").format(num)

def add_idx_suffix(s, idx):
    i = s.rindex('.')
    s = s[:i] + "_" + str(idx) + s[i:]
    return s 

def cv2_image_f2i(img):
    img = (img*255).astype(np.uint8)
    row, col = img.shape
    rate = int(200 / img.shape[0])*1.0
    if rate >= 2:
        img = cv2.resize(img, (int(col*rate), int(row*rate)))
    return img

if __name__=="__main__":
    print(change_suffix("abc.jpg", new_suffix='avi'))