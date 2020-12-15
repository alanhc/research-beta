import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils.files import getBaseName, createFolder
state=''
save = False
def Euclidean_filter(img=None, percent=None,up_threshold=None,lower_threshold=None,  color=None, img_BGR_spilt=None, save_path=None, threshold=None):
    
    filename = save_path.split('/')[-2]
    print(filename)
    if save_path!=None:
        save=True
    
    if color==[255,255,255]:
        state='white'
    elif color==[255,50,150]:
        state='red'
    
    h, w, c = img.shape
    
    r1 = np.full((h, w),color[0])
    g1 = np.full((h, w),color[1])
    b1 = np.full((h, w),color[2])
    

    [b,g,r] = img_BGR_spilt
    img_d = np.sqrt ( (r-r1)**2+(g-g1)**2+(b-b1)**2 )
   
    
    #adaptive_mean = cv2.adaptiveThreshold(img_d, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7,3)
    #adaptive_gaus = cv2.adaptiveThreshold(img_d, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7,3)
    
    #t2, otsu = cv2.threshold(img_d, 0, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    if threshold==None:
        if state == 'red':
            threshold=150
        elif state == 'white':
            threshold=15
    
    
    
    
    


    idx = img_d[:,:] > threshold
    img_filted = np.copy(img)
    img_filted[idx] = 0

    if save:
        img_d = img_d.max() - img_d
        cv2.imwrite(save_path+'_'+state+'_img_distance_norm.png', img_d)
        return img_filted, idx, img_d
        
       
    return img_filted, idx

def binary_color_filter(img):
    img = np.copy(img)
    h,w,c = img.shape
    
    
    colors = [
        [255,0,0],[0,255,0],[0,0,255],
        [255,255,0],[255,0,255],[0,255,255]
    ]
    mask = np.zeros((h,w))
    for color in colors:
        
        lower_color = np.array(color) 
        upper_color = np.array(color) 
        temp = cv2.inRange(img, lower_color, upper_color)
        #mask = np.bitwise_or( np.array(temp, dtype=np.int32), 
        #                      np.array(mask, dtype=np.int32)
        #                    )
        mask = temp + mask
        
    
    #result = cv2.bitwise_and(img, img, mask = mask) 
    return mask