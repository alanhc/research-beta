import pandas as pd
import glob
import cv2
import numpy as np
import json
from utils.files import getBaseName, createFolder
import json

yolo_path = '/home/alanhc-school/yolo/darkflow/'
train_dataset = ['dataset_100', 'pic_100']
for dataset_name in train_dataset:
    base = '../../dataset/'+dataset_name+'/'
    dataset = "origin/"
    save_path = 'img/out/'+dataset_name+'/img_result_light_mask/'
    createFolder('img/out/'+dataset_name+'/img_result_light_mask')
    createFolder('img/out/'+dataset_name+'/img_result_light_mask/yolo')
    createFolder('img/out/'+dataset_name+'/img_result_light_mask/light')
    createFolder('img/out/'+dataset_name+'/img_result_light_mask/combine')
    createFolder('img/out/'+dataset_name+'/img_result_light_mask/yolo_rect')
    createFolder('img/out/'+dataset_name+'/img_result_light_mask/light_rect')
    createFolder('img/out/'+dataset_name+'/img_result_light_mask/combine_rect')
    

    data = pd.read_csv(base+dataset+'data-6.csv')
    test_data = pd.read_csv(base+dataset+'data-6-test.csv')

    print(data.shape)
    print(test_data.shape)

    data = data[['position','fliename', 'answers']]
    test_data = test_data[['predict']]
    data = pd.concat([data,test_data], axis=1)
    print(data.head(3))

    g = data.groupby(["fliename"])


    
    print(data, '===',data)
    print(data.shape)
    print(data.head(3))
    print(g.count().index)


    for filename in g.count().index:
        img = cv2.imread(base+dataset+str(filename)+'.bmp', 1)
        img_yolo = cv2.imread(yolo_path+dataset_name+'_day/'+str(filename)+'.jpg', 1)
        
        img_yolo = cv2.resize(img_yolo, (1920,1080), interpolation = cv2.INTER_AREA)
        img_yolo_rect = np.copy(img_yolo)
        img_combine_rect = np.copy(img)
        img_light_rect = np.copy(img)
        

        img_h, img_w, img_c = img.shape
        light_mask = np.zeros((img_h,img_w,img_c))
        light_mask.fill(255.0)
        #print("group:",g.get_group(filename))
        for box in zip(g.get_group(filename)['position'], g.get_group(filename)['predict']):
            
            if box[1] != 0:
                box = json.loads(box[0]) 
                x, y, w, h = box
                light_mask[y:y+h,x:x+w] = [0,0,0]
                img_light_rect = cv2.rectangle(img_light_rect, (x, y), (x+w, y+h), [0,0,255], 3,cv2.LINE_AA)
        data=[]
        
        with open(yolo_path+'out/'+dataset_name+'_day/yolo_data/'+str(filename)+'.json') as f:
            data = json.load(f)
            data = data.replace("'", '"')
            data = json.loads(data)
            

        yolo_mask = np.zeros((img_h,img_w,img_c))
        combine_mask = np.zeros((img_h,img_w,img_c))
        yolo_mask.fill(255.0)
        combine_mask.fill(255.0)
        
        for d in data:
            
            x = d['topleft']['x']
            y = d['topleft']['y']
            w = d['bottomright']['x']-d['topleft']['x']
            h = d['bottomright']['y']-d['topleft']['y']
            
            yolo_mask[y:y+h,x:x+w] = [0,0,0]
            img_yolo_rect = cv2.rectangle(img_yolo_rect, (x, y), (x+w, y+h), [255,0,0], 3,cv2.LINE_AA)
            

            region_yolo_light = light_mask[y:y+h,x:x+w]
            
            if [0,0,0] in region_yolo_light: # have light region
                combine_mask[y:y+h,x:x+w] = [0,0,0]
                img_combine_rect = cv2.rectangle(img_combine_rect, (x, y), (x+w, y+h), [0,255,0], 3,cv2.LINE_AA)
            
            
            
            
        
        
        
        img = img.astype('uint8')
        img_yolo = img_yolo.astype('uint8')
        yolo_mask = yolo_mask.astype('uint8')
        combine_mask = combine_mask.astype('uint8')
        light_mask = light_mask.astype('uint8')
        
        alpha = 0.5
        beta = (1.0 - alpha)
        img_result_combine = cv2.addWeighted(img, alpha, combine_mask, beta, 0.0)
        img_result_yolo = cv2.addWeighted(img, alpha, yolo_mask, beta, 0.0)
        img_result_light = cv2.addWeighted(img, alpha, light_mask, beta, 0.0)
        
        print(save_path+str(filename)+".png")
        cv2.imwrite(save_path+'combine/'+str(filename)+".png", img_result_combine)
        cv2.imwrite(save_path+'yolo/'+str(filename)+".png", img_result_yolo)
        cv2.imwrite(save_path+'light/'+str(filename)+".png", img_result_light)
        cv2.imwrite(save_path+'combine_rect/'+str(filename)+".png", img_combine_rect)
        cv2.imwrite(save_path+'yolo_rect/'+str(filename)+".png", img_yolo_rect)
        cv2.imwrite(save_path+'light_rect/'+str(filename)+".png", img_light_rect)
        
        
        
        
        #cv2.imshow("vehicle_mask", img_yolo)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        

        
        """
        cv2.imshow("thresh", thresh)
        cv2.imshow("img_dilation", img_dilation)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
            
        
        
        

    