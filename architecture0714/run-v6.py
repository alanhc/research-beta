
import glob
import cv2
import matplotlib.pyplot as plt
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

names = [
            'img', 'img_basic_hsv', 'img_basic_H', 'img_basic_S', 'img_basic_V', 'img_basic_gray', 
            'img_light_detection_gray_th', 'img_light_detection_nakagami_norm', 'img_light_detection_nakagami_norm_th','img_light_detection_nakagami_norm_th_clip',

            'img_color_white_filted', 'img_color_white_mor', 'img_color_white_contour', 
             'img_result_white_edgeboxes', 'img_result_roi_combine'
        ]

createFolder('img')
createFolder('img/out')
for dataset_name in train_dataset:
    createFolder('img/out/'+dataset_name)
    for i in range(len(names)):
            createFolder('img/out/'+dataset_name+'/'+names[i])

def main(frame_path, dataset_name):
    
    print(frame_path)
    filename, f_type = getBaseName(frame_path)
    
    
    #createFolder('img/out/'+dataset_name+'/'+filename)
    #save_path = 'img/out/'+dataset_name+'/'+filename+'/'

    img = cv2.imread(frame_path, 1)
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
    

    """
    tmp = img_nakagami_norm *255.0
    tmp = tmp.astype('uint8') 
    
    
    #ret3,img_nakagami_norm_adaptive = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #print(img_nakagami_norm_adaptive.max())
    #ret2,img_nakagami_norm_adaptive = cv2.threshold(tmp,230,240,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    createFolder('img/out/'+dataset_name+'/'+"img__adaptive")
    cv2.imwrite('img/out/'+dataset_name+'/'+"img__adaptive"+'/'+filename+'.png', img_nakagami_norm_adaptive)
    """
    
    
    
    """
    cv2.imshow("thresh", thresh)
    cv2.imshow("img_dilation", img_dilation)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    

    ### clip center
    img_nakagami_norm_th_clip = clip_center(img_nakagami_norm_th, int(h/3), int(h))
    
    img_white_filted = img_nakagami_norm_th_clip*255.0
    #ret , img_white_filted = cv2.threshold(img_nakagami_norm_th_clip, 245/255.0, 1, cv2.THRESH_TOZERO)
    #img_white_filted = img_white_filted*255.0
    

    k=np.ones((3,3), np.uint8)
    img_white_mor = cv2.morphologyEx(img_white_filted,cv2.MORPH_CLOSE, k,iterations=3) / 255.0
    

    ret, img_white_b = cv2.threshold(img_white_mor, 0.5, 255, cv2.THRESH_BINARY)
    
    ### Contour
    
    img_white_contour = contour_Detection(img_white_b)

    ### EdgeBox
    img_roi_combine = np.copy(img)
    #img_red_edgeboxes,img_roi_combine, boxes_red = Edgeboxes(img_gray=img_red_contour, img_origin=img, color=[0,255,0], img_roi_combine=img_roi_combine, state=state, filename=filename, base=base) #contour, img_origin, box color(BGR)
    img_white_edgeboxes,img_roi_combine, boxes_white = Edgeboxes(img_gray=img_white_contour, img_origin=img, color=[255,0,0], img_roi_combine=img_roi_combine, state=state, filename=filename, base=base) #contour, img_origin, box color(BGR)
    
    ### Fusion
    ### fusion ###
    img_ground = cv2.imread(base+"ground_truth/"+filename+'.bmp',1).astype('uint8')
    img_ground_mask = binary_color_filter(img_ground).astype('uint8')
    
    img_yolo_b = cv2.imread(base+"yolo_binary/"+filename+'.png',0).astype('uint8')
        
    features_white, answers_white = make_feature(boxes_white, version='v6',img_ground=img_ground, img_ground_mask=img_ground_mask, state=state, img_H=img_H, img_yolo_b=img_yolo_b, filename=filename)
        
    features = features_white
    answers = answers_white

    imgs =  [
            img, img_hsv, img_H, img_S, img_V, img_gray*255.0, 
            img_gray_th*255.0, img_nakagami_norm*255.0, img_nakagami_norm_th*255.0, img_nakagami_norm_th_clip*255.0,
            img_white_filted, img_white_mor*255.0, img_white_contour, 
             img_white_edgeboxes, img_roi_combine
    ]
    
    for i in range(len(names)):
        
        save_path = 'img/out/'+dataset_name+'/'+names[i]+'/'+filename
        try:
            cv2.imwrite(save_path+'.png', imgs[i])
        except:
            print('error on ', names[i])
   
    #img_white_edgeboxes,img_roi_combine,
    
    #cv2.imwrite(save_path+'_img_nakagami_thB.png', img_nakagami_thB*255.0)

    #img_white_filted, idx_white, img_white_d = Euclidean_filter(img=img, threshold=20, color=[255,255,255], img_BGR_spilt=[b,g,r], save_path=save_path)
    
    """    
    cv2.imshow("img_nakagami", img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    
    

   
    
    
    

    
    return features, answers

        



    


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
            features = features + f
            answers = answers + a
        data = pd.DataFrame(features, columns=['fliename','iou', 'min', 'std', 'y', 'area', 'position'])
        data['answers'] = pd.DataFrame(answers)
        data.to_csv(base+dataset+'data-6.csv')
        
   