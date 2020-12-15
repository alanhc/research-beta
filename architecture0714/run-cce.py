
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

            'img_color_white_filted', 'img_color_white_filted_gray', 'img_color_white_mor', 'img_color_white_multiply', 
            'img_color_red_filted', 'img_color_red_filted_gray', 'img_color_red_filted_gray_th', 'img_color_red_mor', 
            'img_color_red_mor_th', 'img_color_red_multiply', 'img_color_white_b', 'img_color_red_b', 
            'img_color_white_light', 'img_color_red_light', 'img_color_red_contour', 'img_color_white_contour', 
            'img_result_red_edgeboxes', 'img_result_white_edgeboxes', 'img_result_roi_combine'
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
    
    ret, img_nakagami_norm_th = cv2.threshold(img_nakagami_norm, 0.88, 1, cv2.THRESH_TOZERO)

    ### Area filter
    thresh = (img_nakagami_norm_th * 255.0).astype('uint8')
    
    img_ccl_origin, img_ccl_show, label_nums = ccl(thresh)    ### need imwrite
    
    
    """
    cv2.imshow("thresh", thresh)
    cv2.imshow("opening", img_ccl_show)
    #cv2.imshow("sure_bg", sure_bg)
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    """
    for (i, label) in enumerate(np.unique(img_ccl_origin)):
        if label == 0: #background
            continue
        else:
            labelMask = np.zeros(img_nakagami_norm_th_tmp.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)
    """

    

    ### clip center
    img_nakagami_norm_th_clip = clip_center(img_nakagami_norm_th, int(h/3), int(h))
    #print(img_nakagami_norm_th_center.max())
    #print(img_nakagami_norm_th.max())
    ### Color filter
    #img_white_filted, idx_white, img_white_d = Euclidean_filter(img=img, threshold=15, color=[255,255,255], img_BGR_spilt=[b,g,r], save_path=save_path)
    ret , img_white_filted = cv2.threshold(img_gray, 245/255.0, 1, cv2.THRESH_TOZERO)
    img_white_filted = img_white_filted*255.0
    #img_red_filted, idx_red, img_red_d = Euclidean_filter(img=img, threshold=150, color=[255,50,150], img_BGR_spilt=[b,g,r], save_path=save_path)
    


    lower_red = np.array([0,100,5])
    upper_red = np.array([15,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    
    lower_red = np.array([240,100,5])
    upper_red = np.array([255,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    mask = mask0 + mask1
    img_red_filted = img.copy()
    img_red_filted[np.where(mask==0)] = 0
    
    
    cv2.imshow("mask0", mask0)
    cv2.imshow("mask1", mask1)
    cv2.imshow("mask", mask)
    #cv2.imshow("sure_bg", sure_bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    ###### 形態學操作
    #img_white_filted_gray = cv2.cvtColor(img_white_filted, cv2.COLOR_BGR2GRAY)
    #img_white_filted_gray = img_white_filted
    k=np.ones((3,3), np.uint8)
    img_white_mor = cv2.morphologyEx(img_white_filted,cv2.MORPH_CLOSE, k,iterations=3) / 255.0
    

    
    img_red_filted_gray = cv2.cvtColor(img_red_filted, cv2.COLOR_BGR2GRAY)
    ret , img_red_filted_gray_th = cv2.threshold(img_red_filted_gray, 150, 255, cv2.THRESH_TOZERO)
    k=np.ones((13,13), np.uint8)
    #k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
    img_red_mor = cv2.morphologyEx(img_red_filted_gray,cv2.MORPH_CLOSE, k,iterations=5)
    ret, img_red_mor_th = cv2.threshold(img_red_mor, 80, 255, cv2.THRESH_TOZERO)
    

    ### Multiply color and light
    img_white_multiply = img_nakagami_norm_th_clip * img_white_mor
    img_red_multiply = img_nakagami_norm_th_clip * img_red_mor_th
    
    ret, img_white_b = cv2.threshold(img_white_multiply, 0.5, 255, cv2.THRESH_BINARY)
    ret, img_red_b = cv2.threshold(img_red_multiply, 0.5, 255, cv2.THRESH_BINARY)

    img_white_light = img_white_b - img_red_b
    img_red_light = img_red_b

    img_white_light = np.clip(img_white_light,0,255)
    img_red_light = np.clip(img_red_light,0,255)
    
    

    ### Contour
    img_red_contour = contour_Detection(img_red_light)
    img_white_contour = contour_Detection(img_white_light)

    ### EdgeBox
    img_roi_combine = np.copy(img)
    img_red_edgeboxes,img_roi_combine, boxes_red = Edgeboxes(img_gray=img_red_contour, img_origin=img, color=[0,255,0], img_roi_combine=img_roi_combine, state=state, filename=filename, base=base) #contour, img_origin, box color(BGR)
    img_white_edgeboxes,img_roi_combine, boxes_white = Edgeboxes(img_gray=img_white_contour, img_origin=img, color=[255,0,0], img_roi_combine=img_red_edgeboxes, state=state, filename=filename, base=base) #contour, img_origin, box color(BGR)
    
    ### Fusion
    ### fusion ###
    img_ground = cv2.imread(base+"ground_truth/"+filename+'.bmp',1).astype('uint8')
    img_ground_mask = binary_color_filter(img_ground).astype('uint8')
    
    img_yolo_b = cv2.imread(base+"yolo_binary/"+filename+'.png',0).astype('uint8')
        
    features_red, answers_red = make_feature(boxes_red, img_ground=img_ground, img_ground_mask=img_ground_mask, state=state, img_H=img_H, img_yolo_b=img_yolo_b, filename=filename)
    features_white, answers_white = make_feature(boxes_white, img_ground=img_ground, img_ground_mask=img_ground_mask, state=state, img_H=img_H, img_yolo_b=img_yolo_b, filename=filename)
        
    features = features_red + features_white
    answers = answers_red + answers_white

    imgs =  [
            img, img_hsv, img_H, img_S, img_V, img_gray*255.0, 
            img_gray_th*255.0, img_nakagami_norm*255.0, img_nakagami_norm_th*255.0, img_nakagami_norm_th_clip*255.0,
            img_white_filted, img_white_filted_gray, img_white_mor*255.0, img_white_multiply*255.0, 
            img_red_filted, img_red_filted_gray, img_red_filted_gray_th, img_red_mor, 
            img_red_mor_th, img_red_multiply, img_white_b, img_red_b, 
            img_white_light, img_red_light, img_red_contour, img_white_contour, 
            img_red_edgeboxes, img_white_edgeboxes, img_roi_combine
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
    """
    ### 形態學操作
    #r1=cv2.morphologyEx(img_red,cv2.MORPH_CLOSE, k,iterations=3)
    #k=np.ones((3,3), np.uint8)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    img_mor = cv2.morphologyEx(img_red, cv2.MORPH_CLOSE, k,iterations=3)

    img_ccl_origin, img_ccl_show, label_num = ccl(img_mor)
    img_red_box, red_boxes = find_BoundingBox(img_ccl_origin, img)
    """
    """
    cv2.imshow("img_red_filled", img_box)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    

   
    

    """
    cv2.imshow("img_red", img_mor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    

    
    

   
    
    
    

    
    
    
    """
    h, w, c = img.shape
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_H = img_hsv[:,:,0]
    img_S = img_hsv[:,:,1]
    img_V = img_hsv[:,:,2]
    
    [b,g,r] = cv2.split(img)
    
    
    ### image enhacement
    img_th = color_intensity(img)
    img_nakagami = Nakagami_image_enhancement(img_th, 3)
    ret, img_nakagami_thB = cv2.threshold(img_nakagami, 100, 1, cv2.THRESH_BINARY)

    img_red_filted, idx_red = Euclidean_filter(img, 1,150,0,[255,0,0], [b,g,r], save_path)  #img, percent,up_threshold,lower_threshold color(RGB), save_path
    img_white_filted,idx_white = Euclidean_filter(img, 1,255,7, [255,255,255], [b,g,r], save_path)  #img, percent, color(RGB), save_path
    
    
    img_red_filted_gray = cv2.cvtColor(img_red_filted, cv2.COLOR_BGR2GRAY)
    img_white_filted_gray = cv2.cvtColor(img_white_filted, cv2.COLOR_BGR2GRAY)
    
    img_red_filted_gray_max = img_red_filted_gray.max()
    img_white_filted_gray_max = img_white_filted_gray.max()
    

    img_red_filted_gray_norm = img_red_filted_gray/img_red_filted_gray_max
    img_white_filted_gray_norm = img_white_filted_gray/img_white_filted_gray_max

    
    img_red_multiply = img_nakagami_thB * img_red_filted_gray_norm
    img_white_multiply = img_nakagami_thB * img_white_filted_gray_norm

    img_red_multiply = img_red_multiply*255.0
    img_white_multiply = img_white_multiply*255.0
    
    #img_red_nakagami = Nakagami_image_enhancement(img_red_filted_gray_norm, 3) #gray_img, kernel_size
    #img_white_nakagami = Nakagami_image_enhancement(img_white_filted_gray_norm, 3) #gray_img, kernel_size
    
    
    img_red_nakagami_cliped = clip_center(img_red_multiply, int(h/3), int(h)), # img, y_up, y_down
    img_white_nakagami_cliped = clip_center(img_white_multiply, int(h/3), int(h)), # img, y_up, y_down
    img_red_nakagami_cliped = img_red_nakagami_cliped[0]
    img_white_nakagami_cliped = img_white_nakagami_cliped[0]
   
    
    

    img_red_contour = contour_Detection(img_red_nakagami_cliped)
    img_white_contour = contour_Detection(img_white_nakagami_cliped)

    
    
    
    #print(type(img_red_contour_cliped[0]))
    """
    """
    cv2.imshow("img_red_contour_cliped", img_red_contour_cliped)
    #cv2.imshow("img_white_filted_gray", img_white_filted_gray)
    #cv2.imshow("img_red_nakagami", img_red_nakagami)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    """
    img_roi_combine = np.copy(img)
    img_red_edgeboxes,img_roi_combine, boxes_red = Edgeboxes(img_red_contour, img, [0,255,0], img_roi_combine, state=state, filename=filename, base=base) #contour, img_origin, box color(BGR)
    img_white_edgeboxes,img_roi_combine, boxes_white = Edgeboxes(img_white_contour, img, [255,0,0], img_red_edgeboxes, state=state, filename=filename, base=base) #contour, img_origin, box color(BGR)
    
    
    ### fusion ###
    
    

    img_ground = cv2.imread(base+"ground_truth/"+filename+'.bmp',1).astype('uint8')
    img_ground_mask = binary_color_filter(img_ground).astype('uint8')
    
    img_yolo_b = cv2.imread(base+"yolo_binary/"+filename+'.png',0).astype('uint8')
        
    features_red, answers_red = make_feature(boxes_red, img_ground=img_ground, img_ground_mask=img_ground_mask, state=state, img_H=img_H, img_yolo_b=img_yolo_b, filename=filename)
    features_white, answers_white = make_feature(boxes_white, img_ground=img_ground, img_ground_mask=img_ground_mask, state=state, img_H=img_H, img_yolo_b=img_yolo_b, filename=filename)
        
    features = features_red + features_white
    answers = answers_red + answers_white
        
       
    
        
    ### report ###
    
        

    if save:
        cv2.imwrite(save_path+'_origin.png', img)
        cv2.imwrite(save_path+'_color_intensity_with_threshold.png', img_th*255.0)
        cv2.imwrite(save_path+'_img_nakagami.png', img_nakagami)
        cv2.imwrite(save_path+'_img_nakagami_thB.png', img_nakagami_thB*255.0)
        
        
        
        cv2.imwrite(save_path+'_img_H.png', img_H)
        cv2.imwrite(save_path+'_img_S.png', img_S)
        cv2.imwrite(save_path+'_img_V.png', img_V)
        cv2.imwrite(save_path+'_img_red_filted.png', img_red_filted)
        cv2.imwrite(save_path+'_img_white_filted.png', img_white_filted)
        cv2.imwrite(save_path+'_img_red_filted_gray.png', img_red_filted_gray)
        
        #cv2.imwrite(save_path+'_img_red_nakagami.png', img_red_nakagami)
        #cv2.imwrite(save_path+'_img_white_nakagami.png', img_white_nakagami)
        
        cv2.imwrite(save_path+'_img_red_nakagami_cliped.png', img_red_nakagami_cliped)
        cv2.imwrite(save_path+'_img_white_nakagami_cliped.png', img_white_nakagami_cliped)

        cv2.imwrite(save_path+'_img_red_multiply.png', img_red_multiply)
        cv2.imwrite(save_path+'_img_white_multiply.png', img_white_multiply)
        
        
        cv2.imwrite(save_path+'_img_red_contour.png', img_red_contour)
        cv2.imwrite(save_path+'_img_white_contour.png', img_white_contour)

        
        

        cv2.imwrite(save_path+'_img_red_edgeboxes.png', img_red_edgeboxes)
        cv2.imwrite(save_path+'_img_white_edgeboxes.png', img_white_edgeboxes)
        cv2.imwrite(save_path+'_img_roi_combine.png', img_roi_combine)

    

        #cv2.imwrite(save_path+'_img_red.png', r)
    return features, answers
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
        data.to_csv(base+dataset+'data-5.csv')
        
   