import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils.files import getBaseName, createFolder
state=''
def Euclidean_filter(img, percent,up_threshold,lower_threshold,  color, img_BGR_spilt, save_path):
    filename = save_path.split('/')[2]
   
    if color==[255,255,255]:
        state='white'
    elif color==[255,0,0]:
        state='red'
    
    h, w, c = img.shape
    r1 = np.full((h, w),color[0])
    g1 = np.full((h, w),color[1])
    b1 = np.full((h, w),color[2])
    
    [b,g,r] = img_BGR_spilt
    distance = np.sqrt ( (r-r1)**2+(g-g1)**2+(b-b1)**2 )
    """
    plt.hist(distance, bins = 10)
    plt.title(filename+' '+state+" histogram") 
    plt.xlabel("value")
    plt.ylabel("Frequency")
    plt.savefig(save_path+filename+'_'+state+'_histogram.jpg')
    
    """
    print('=== '+state+' distance max ===',distance.max())
    plt.hist(distance, bins = [0,20,40,60,80,100]) 
    plt.title("histogram") 
    plt.xlabel("value")
    plt.ylabel("Frequency")
    plt.savefig(save_path+state+'_histogram.jpg')
    """

    threshold = np.percentile(distance, percent)
    #threshold = 100
    if threshold > up_threshold:
        threshold = up_threshold
    if threshold < lower_threshold:
        threshold = lower_threshold
    print('=== threshold:'+str(threshold)+' ===')

    idx = distance[:,:] > threshold
    img_filted = np.copy(img)
    img_filted[idx] = 0

    
    return img_filted, idx
