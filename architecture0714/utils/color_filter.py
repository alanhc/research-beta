import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils.files import getBaseName, createFolder
state=''
save = False
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
    d_max = distance.max()
    d_min = distance.min()
    d = (d_max-d_min)
    img_d = ((distance-d_min)/d*255.0).astype('uint8')
    
    
    if state == 'red':
        threshold=60
    elif state == 'white':
        threshold=15
    
    #cv2.imshow("img_d", img_d)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    print(filename)


    idx = img_d[:,:] > threshold
    img_filted = np.copy(img)
    img_filted[idx] = 0
    if save:
        cv2.imwrite('img/out/'+filename+'/'+filename+'_'+state+'_img_distance_norm.png', img_d)
        cv2.imwrite('img/out/'+filename+'/'+filename+'_'+state+'_img_adaptive_gaus.png', adaptive_gaus)
        cv2.imwrite('img/out/'+filename+'/'+filename+'_'+state+'_img_adaptive_mean.png', adaptive_mean)
        
        cv2.imwrite('img/out/'+filename+'/'+filename+'_'+state+'_img_otsu.png', otsu)
        
       
    
    return img_filted, idx
