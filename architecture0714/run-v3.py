
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
########## Method import ##########
from utils.files import getBaseName, createFolder
from utils.color_filter import Euclidean_filter
from utils.symbolic import symbolic_image
from utils.connect_compoent import ccl, find_BoundingBox

base = '../dataset_100/dataset_100/'

save = True

def main(frame_path):
    
    filename, f_type = getBaseName(frame_path)
    createFolder('img')
    createFolder('img/out')
    createFolder('img/out/'+filename)
    save_path = 'img/out/'+filename+'/'

    img = cv2.imread(frame_path, 1)
    h, w, c = img.shape
    [b,g,r] = cv2.split(img)
    img_red_filted, idx_red = Euclidean_filter(img, 1,150,0,[255,0,0], [b,g,r], save_path)
    img_white_filted, idx_white = Euclidean_filter(img, 1,255,7, [255,255,255], [b,g,r], save_path)

    img_symbolic_idx, img_symbolic_show  = symbolic_image([idx_red,idx_white], (h,w))
    
    
    img_light_b = np.where(img_symbolic_idx>0, 255, 0).astype('uint8')
    
    
    img_ccl_origin, img_ccl_show, label_num = ccl(img_light_b)
    img_box, boxes = find_BoundingBox(img_ccl_origin, img)
    
    """
    cv2.imshow("img_ccl", img_ccl)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    
    #print(frame_path,label_num, img_ccl_origin.shape)
    
    if save:
        cv2.imwrite('img/out/'+filename+'/'+filename+'_origin.png', img)
        cv2.imwrite('img/out/'+filename+'/'+filename+'_img_red_filted.png', img_red_filted)
        cv2.imwrite('img/out/'+filename+'/'+filename+'_img_white_filted.png', img_white_filted)
        
        cv2.imwrite('img/out/'+filename+'/'+filename+'_img_symbolic.png', img_symbolic_show)
        cv2.imwrite('img/out/'+filename+'/'+filename+'_img_light_b.png', img_light_b)
        cv2.imwrite('img/out/'+filename+'/'+filename+'_img_ccl.png', img_ccl_show)        
        cv2.imwrite('img/out/'+filename+'/'+filename+'_img_box.png', img_box)
        
        
        

    



    

if __name__ == '__main__':
    files = glob.glob(base+"origin-small/*")
    for f in sorted(files):
        main(f)
   