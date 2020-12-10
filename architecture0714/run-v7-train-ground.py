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

from utils.connect_compoent import *
from utils.symbolic import symbolic_image
from utils.connect_compoent import ccl, find_BoundingBox

from utils.train_model import *

train_dataset = ['dataset_100', 'pic_100', 'fewer_light_100']
#train_dataset = ['dataset_100', 'pic_100']
#train_dataset = ['fewer_light_100']

dataset = "origin/"
#dataset = "origin-small/"


def make_training_data(base, frame_path, dataset_name):
    
    print(frame_path)
    filename, f_type = getBaseName(frame_path)
    
    img = cv2.imread(base+dataset+"/"+filename+'.bmp',1).astype('uint8')
    img_ground = cv2.imread(base+"ground_truth/"+filename+'.bmp',1).astype('uint8')
    img_yolo_b = cv2.imread(base+"yolo_binary/"+filename+'.png',0).astype('uint8')
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    img_hsv[:,:,0] = img_hsv[:,:,0] / img_hsv[:,:,0].max() * 255.0
    img_hsv[:,:,0] = img_hsv[:,:,1] / img_hsv[:,:,1].max() * 255.0
    img_hsv[:,:,0] = img_hsv[:,:,2] / img_hsv[:,:,2].max() * 255.0
    
    img_H = img_hsv[:,:,0]
    img_S = img_hsv[:,:,1]
    img_V = img_hsv[:,:,2]

    img_ground_mask = binary_color_filter(img_ground).astype('uint8')
    
    img_ccl_origin, img_ccl_show, label_num = ccl(img_ground_mask)
    img_box, boxes = find_BoundingBox(img_ccl_origin, img)

    features = []
    answers = []
    for b in boxes:
        x = b[0]
        y = b[1]
        w = b[2] - b[0]
        h = b[3] - b[1]
        
        ### make feature
        
        
        yolo_and_edgebox = img_yolo_b[y:y+h,x:x+w]
        area_yolo_and_edgebox = (yolo_and_edgebox//255).sum()

        ROI_combine = area_yolo_and_edgebox/(w*h)
        center_line = img_S[y+h//2, x:x+w]
        ROI_center_std = np.std(center_line)
        ROI_center_min = np.amin(center_line)
        ROI_center_area = w*h
        ROI_height = y
        
        feature = [filename,ROI_combine,ROI_center_min,ROI_center_std,ROI_height,ROI_center_area, [x,y,w,h]]


        #print(feature)
        ### make answer

        yolo_and_ground = np.bitwise_and(img_ground_mask[y:y+h,x:x+w] ,img_yolo_b[y:y+h,x:x+w])
        area_yolo_and_ground = (yolo_and_ground//255).sum()

        img_ROI = img_ground[y:y+h,x:x+w]
        t ,ct= np.unique(img_ROI.reshape(-1, img_ROI.shape[2]), axis=0, return_counts=True)
        #print(t,ct)
        
        answer = ""
        #if len(t)==1:
        t = t.tolist()
        idx, = np.where(ct == ct.max()) 
        if t[int(idx)] in [[0,0,255], [0,0,0]]:
            answer=0
        else:
            answer=1
        
        
        
        
        
        if answer not in [0,1]:
            if t in [[0,0,255], [0,0,0]]:
                answer=0
            else:
                answer=1

            cv2.imshow("img_ROI", img_ROI)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        """
        print("iou:", ROI_combine, "answer:", answer)
        #cv2.imshow("img_ground", img_ground)
        cv2.imshow("img_ROI", img_ROI)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        
        features.append(feature)
        answers.append(answer)
    

            
            
            
        
        
        
        

    #print(boxes)
    
    
    """
    cv2.imshow("img_ground", img_box)
    cv2.imshow("img_ground_mask", img_ground_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    return features,answers

if __name__ == '__main__':
    
    for dataset_name in train_dataset:
        
        base = '../../dataset/'+dataset_name+'/'
        files = glob.glob(base+dataset+'*.bmp')
        features=[]
        answers=[]
        ### make training data
        i=1
        for f_name in sorted(files):
            tStart = time.time()
            f, a = make_training_data(base, f_name, dataset_name)
            tEnd = time.time()
            features.extend(f)
            answers.extend(a)
            
            print("It cost %f sec"% (tEnd - tStart))
            print("remain:",(len(files)-i)*(tEnd - tStart), "sec")
            i+=1
        
        data = pd.DataFrame(features, columns=['fliename','iou', 'min', 'std', 'y', 'area', 'position'])
        data['answers'] = pd.DataFrame(answers)
        print("==========", base+dataset+'data-7-train-'+dataset.split('/')[0]+'.csv')
        data.to_csv(base+dataset+'data-7-train-'+dataset.split('/')[0]+'.csv')
        
        #training
        all_data = pd.read_csv(base+dataset+'data-7-train-'+dataset.split('/')[0]+'.csv')
        print('all_data:',all_data.shape)
        all_count = all_data.answers.value_counts().sort_index()
        max_count = all_count.max()
        len_data = len(all_count) 
        print(all_count)

        df_class=[]
        for i in range(len_data):
            idx=all_count.index[i]
            print(i)
            df_class.append(all_data[all_data['answers'] == idx])

        df_test_over = pd.DataFrame()
        #print('===',all_count[2])
        for i in range(len_data):
            
            idx=all_count.index[i]
            if all_count[idx]!= max_count:
                df_class_over = df_class[i].sample(max_count, replace=True)
                df_test_over = pd.concat([df_test_over,df_class_over], axis=0)
            else:
                df_test_over = pd.concat([df_test_over,df_class[i]], axis=0)

        print(df_test_over.answers.value_counts())

        data = df_test_over
        X_train = data[['iou', 'min', 'std', 'y', 'area']]
        #X_train = data[['iou', 'min', 'std']]
        y_train = data['answers']
        print("X_train:",X_train.shape, "X_train:", y_train.shape)
        
        
        print("=====Training models=====")
        save_path = base+dataset+'data-7-train-'+dataset.split('/')[0]
        rf(X_train=X_train, y_train=y_train, save_path=save_path+'-model-rf.pkl' )
        svm(X_train=X_train, y_train=y_train, save_path=save_path+'-model-svm.pkl' )
         
       