import numpy as np
import cv2
from math import gamma
from math import exp

def Nakagami_image_enhancement(img, kernal_size):
    img_r2=img**2
    kernel = np.ones((kernal_size,kernal_size),np.float32)/kernal_size**2
    omega = cv2.filter2D(img_r2,-1,kernel) # src, dst, kernel
    omega2 = omega**2

    img_process = (img_r2-omega)**2
    dst = cv2.filter2D(img_process,-1,kernel)
    m=omega2/(dst+0.01)

    X = img
    h, w = X.shape
    img_nakagami = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            Mij = m[i,j]
            OMEGAij = omega[i,j]
            Xij = X[i,j]
            if Mij>=0.5 and OMEGAij>0:        
                #img_nakagami[i,j] = (2*Mij**Mij / (gamma(Mij)*(OMEGAij**Mij)) ) * (Xij**(2*Mij-1)) * exp(-(Mij*(Xij**2))/OMEGAij)
                img_nakagami[i,j] = (2*Mij**Mij / (gamma(Mij)*(OMEGAij**Mij)) ) * (Xij**(2*Mij-1)) * exp(-(Mij*(Xij**2))/OMEGAij)

    img_max = np.max(img_nakagami)
    img_min = np.min(img_nakagami)
    
    img_nakagami = img_nakagami/img_max*255.0
    #print('============== nakagami  =============',img_nakagami.max(), img_nakagami.min())
    return img_nakagami

def contour_Detection(img):
    img = np.array(img,np.uint8)
    image, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape
    img_contour = np.zeros((h,w))
    img_contour = cv2.drawContours(img_contour, contours, -1, 255, 1)

    return img_contour

def clip_center(img, y_up, y_down):
    
    img_tmp = np.zeros(img.shape).astype('uint8')
    img_tmp[y_up:y_down,:] = img[y_up:y_down,:]
    
    return img_tmp

def color_intensity_th(img):
    max_img = np.max(a=img, axis=2)
    img_c = max_img / 255.0
    ret , img_threshold = cv2.threshold(img_c, 0.4, 1, cv2.THRESH_TOZERO)
    return img_threshold*255.0