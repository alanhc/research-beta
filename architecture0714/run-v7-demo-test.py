
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import math
########## Method import ##########
from utils.files import getBaseName, createFolder
from utils.color_filter import Euclidean_filter, binary_color_filter
from utils.enhancement import Nakagami_image_enhancement, contour_Detection, clip_center, color_intensity
from utils.edgeBoxes import Edgeboxes
from utils.fusion import make_feature
from utils.connect_compoent import *
from utils.symbolic import symbolic_image

names = [
            'img', 'img_basic_hsv', 'img_basic_H', 'img_basic_S', 'img_basic_V', 'img_basic_gray', 
            'img_light_detection_gray_th', 'img_light_detection_nakagami_norm', 'img_light_detection_nakagami_norm_th','img_light_detection_nakagami_norm_th_clip',

            'img_color_white_filted', 'img_color_white_mor', 'img_color_white_binary','img_color_white_contour', 
             'img_result_white_edgeboxes', 'img_result_roi_combine'
        ]

datasets = glob.glob('img/demo-out/3/origin/*') 
for i in range(len(names)):
    createFolder('img/demo-out/3/'+names[i])

save = True
nakagami = True
state='test'
yolo_b_path = '/home/alanhc-school/yolo/darkflow/out/tlchia-dataset-v2_day/binary/'

def main(frame_name):
    features=""
    print(frame_name)
    filename, f_type = getBaseName(frame_name)
    
    img = cv2.imread(frame_name, 1)
    h, w, c = img.shape
    [b,g,r] = cv2.split(img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    img_hsv[:,:,0] = img_hsv[:,:,0] / img_hsv[:,:,0].max() * 255.0
    img_hsv[:,:,0] = img_hsv[:,:,1] / img_hsv[:,:,1].max() * 255.0
    img_hsv[:,:,0] = img_hsv[:,:,2] / img_hsv[:,:,2].max() * 255.0
    
    img_H = img_hsv[:,:,0]
    img_S = img_hsv[:,:,1]
    img_V = img_hsv[:,:,2]

    ### Light Detection
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    ret , img_gray_th = cv2.threshold(img_gray, 0.4, 1, cv2.THRESH_TOZERO)
    

    if nakagami:
        img_nakagami = Nakagami_image_enhancement(img_gray_th, 3)
    else:
        img_nakagami = img_gray_th
    img_nakagami_norm = img_nakagami/img_nakagami.max() # resize to [0,1]
    
    
    
    ret, img_nakagami_norm_th = cv2.threshold(img_nakagami_norm, 200/255, 1, cv2.THRESH_TOZERO)
    

   
    

    ### clip center
    img_nakagami_norm_th_clip = clip_center(img_nakagami_norm_th, int(h/3), int(h))
    
    img_white_filted = img_nakagami_norm_th_clip*255.0
    ret, img_white_filted = cv2.threshold(img_white_filted, 230, 255, cv2.THRESH_TOZERO)
    #ret , img_white_filted = cv2.threshold(img_nakagami_norm_th_clip, 245/255.0, 1, cv2.THRESH_TOZERO)
    #img_white_filted = img_white_filted*255.0
    

    k=np.ones((3,3), np.uint8)
    img_white_mor = cv2.morphologyEx(img_white_filted,cv2.MORPH_CLOSE, k,iterations=3) / 255.0
    

    ret, img_white_b = cv2.threshold(img_white_mor, 0.9, 255, cv2.THRESH_BINARY)
    
    ### Contour
    img_white_contour = contour_Detection(img_white_b)


    
    ### EdgeBox
    img_roi_combine = np.copy(img)
    #img_red_edgeboxes,img_roi_combine, boxes_red = Edgeboxes(img_gray=img_red_contour, img_origin=img, color=[0,255,0], img_roi_combine=img_roi_combine, state=state, filename=filename, base=base) #contour, img_origin, box color(BGR)
    img_white_edgeboxes,img_roi_combine, boxes_white = Edgeboxes(img_gray=img_white_contour, img_origin=img, color=[255,0,0], img_roi_combine=img_roi_combine, state=state, filename=filename, base=None) #contour, img_origin, box color(BGR)
    
    img_yolo_b = cv2.imread(yolo_b_path+filename+'.png',0).astype('uint8')
    features_white = make_feature(boxes=boxes_white, version='v6',img_ground=None, img_ground_mask=None, state='test', img_S=img_S, img_yolo_b=img_yolo_b, filename=filename)
    
    features = features_white
    ### save files ### 
    imgs =  [
            img, img_hsv, img_H, img_S, img_V, img_gray*255.0, 
            img_gray_th*255.0, img_nakagami_norm*255.0, img_nakagami_norm_th*255.0, img_nakagami_norm_th_clip*255.0,
            img_white_filted, img_white_mor*255.0, img_white_b, img_white_contour, 
            img_white_edgeboxes, img_roi_combine
    ]

    dataset_name = frame_name.split('/')[2]
    for i in range(len(names)):
        
        save_path = 'img/demo-out/'+dataset_name+'/'+names[i]+'/'+filename
        try:
            cv2.imwrite(save_path+'.png', imgs[i])
        except:
            print('error on ', names[i])

    return features

if __name__ == '__main__':
    print(len(datasets))
    features = []
    i=1
    for f in datasets:
        tStart = time.time()
        f = main(f)
        tEnd = time.time()
        print("It cost %f sec"% (tEnd - tStart))
        print("remain:",(len(datasets)-i)*(tEnd - tStart), "sec")
        features = features + f
        i+=1
    data = pd.DataFrame(features, columns=['fliename','iou', 'min', 'std', 'y', 'area', 'position'])
    data.to_csv('./img/demo-out/data-test.csv')
