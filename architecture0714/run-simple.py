
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import time
########## Method import ##########
from utils.files import getBaseName, createFolder
from utils.color_filter import Euclidean_filter, binary_color_filter
from utils.enhancement import Nakagami_image_enhancement, contour_Detection, clip_center, color_intensity
from utils.edgeBoxes import Edgeboxes
from utils.fusion import make_feature
from utils.connect_compoent import *
from utils.symbolic import symbolic_image

train_dataset = ['dataset_100', 'pic_100']
#train_dataset = ['pic_100']

dataset = "origin/"

save = True
state = 'train'
nakagami = True
def BGR_to_RGB(img):
    if len(img.shape)==3:
        h, w, c = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    else:
        return img
    
    


def main(frame_path, dataset_name):
    print(frame_path)
    filename, f_type = getBaseName(frame_path)
    createFolder('img')
    createFolder('img/out-simple')
    createFolder('img/out-simple/'+dataset_name)
    save_path = 'img/out/'+dataset_name+'/simple/'
    createFolder(save_path)

    img = cv2.imread(frame_path, 1)
    h, w, c = img.shape
    [b,g,r] = cv2.split(img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_H = img_hsv[:,:,0]
    img_S = img_hsv[:,:,1]
    img_V = img_hsv[:,:,2]
    img_color_intensity = color_intensity(img)
    

    ### Light Detection
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    ret , img_gray = cv2.threshold(img_gray, 0.4, 1, cv2.THRESH_TOZERO)
    #ret , img_gray_th = cv2.threshold(img_gray, 245/255, 1, cv2.THRESH_TOZERO)
    #ret, img_gray_th = cv2.threshold(img_gray, 245/255, 1, cv2.THRESH_TOZERO)
    #ret, img_color_intensity_th = cv2.threshold(img_color_intensity, 0.2, 1, cv2.THRESH_TOZERO)

    img_nakagami = Nakagami_image_enhancement(img_gray, 5) / 255.0
    img_nakagami_c = Nakagami_image_enhancement(img_color_intensity, 5) / 255.0
    
    
    
    ret, img_nakagami_norm_th = cv2.threshold(img_nakagami, 240/255, 1, cv2.THRESH_TOZERO)
    ret, img_nakagami_c_th = cv2.threshold(img_nakagami_c, 240/255, 1, cv2.THRESH_TOZERO)


    

    pipline = [ 'img','img_H','img_S','img_V','img_gray','img_color_intensity',
                'img_nakagami', 'img_nakagami_c','img_nakagami_norm_th', 'img_nakagami_c_th'
                ]
    imgs = [img,img_H,img_S,img_V,img_gray,img_color_intensity,img_nakagami, img_nakagami_c,  img_nakagami_norm_th,img_nakagami_c_th]

    fig = plt.figure( figsize=(32,64))
    
    for i in range(len(imgs)):
        ax = fig.add_subplot(len(imgs)//2+1, 2, i+1)
        img_tmp = BGR_to_RGB(imgs[i])
        ax.set_title(pipline[i])
        if pipline[i] in ['img_gray_th_','img_nakagami_']:
            plt.imshow(img_tmp) 
            plt.colorbar()
        elif len(img_tmp.shape)==2:
            plt.imshow(img_tmp, cmap='gray') 
        else:
            plt.imshow(img_tmp) 
        
    plt.savefig(save_path+dataset_name+'_'+filename+'.png', dpi=200)
        
        

    

    
    return -1, -1

        



    

if __name__ == '__main__':
    for dataset_name in train_dataset:
        base = '../../dataset/'+dataset_name+'/'
        files = glob.glob(base+dataset+'*.bmp')
        features=[]
        answers=[]
        for f_name in sorted(files):
            tStart = time.time()
            f, a = main(f_name, dataset_name)
            tEnd = time.time()
            print("It cost %f sec"% (tEnd - tStart))
           
           
    
        
   