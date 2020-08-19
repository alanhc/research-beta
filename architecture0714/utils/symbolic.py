import numpy as np
import cv2

def symbolic_image(idxs, shape):
    (h,w) = shape
    
    img_symbolic_idx = np.zeros(shape)
    i=1
    for idx in idxs:
        idx = np.invert(idx)
        img_symbolic_idx[idx] = i
        i+=1
    img_symbolic_show = np.copy(img_symbolic_idx)
    img_symbolic_show = np.where(img_symbolic_idx==1, 127, img_symbolic_show) 
    img_symbolic_show = np.where(img_symbolic_idx==2, 255, img_symbolic_show) 

    label_hue = np.uint8(179*img_symbolic_show/np.max(img_symbolic_show))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0
    return img_symbolic_idx, labeled_img