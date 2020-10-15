import pandas as pd
from joblib import dump, load
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from utils.files import getBaseName, createFolder
import cv2
import numpy as np
import json

createFolder('img/demo-out/3/img_result_light_mask')
names = ['yolo', 'yolo_rect', 'light', 'light_rect', 'combine', 'combine_rect']
for i in range(len(names)):
    createFolder('img/demo-out/3/img_result_light_mask/'+names[i])

dataset = "origin/"

yolo_path = '/home/alanhc-school/yolo/darkflow/'


model_path = '/home/alanhc-school/Downloads/research/dataset/dataset_100/origin/data-7-train-origin-model-rf.pkl'

test_data_path = '/home/alanhc-school/Downloads/research/research-beta/architecture0714/img/demo-out/data-test.csv'
predict_data_path = '/home/alanhc-school/Downloads/research/research-beta/architecture0714/img/demo-out/data-test-predict.csv'


model = load(model_path)

test_data = pd.read_csv(test_data_path)

print(test_data.shape)

X_test = test_data[['iou', 'min', 'std', 'y', 'area']]


y_pred = model.predict(X_test)
y_pred = pd.DataFrame({'predict':y_pred})
y_pred.to_csv(predict_data_path)

data = pd.concat([y_pred,test_data], axis=1)
print(data.head(3))
g = data.groupby(["fliename"])
print(g.count().index)



img_path = '/home/alanhc-school/Downloads/research/research-beta/architecture0714/img/demo-out/3/img/'
img_yolo_path = '/home/alanhc-school/yolo/darkflow/tlchia-dataset-v2_day/'
for filename in g.count().index:
    img = cv2.imread(img_path+str(filename)+'.png', 1)
    img_yolo = cv2.imread(img_yolo_path+str(filename)+'.jpg', 1)
    img_yolo = cv2.resize(img_yolo, (1920,1080), interpolation = cv2.INTER_AREA)
    
    img_yolo_rect = np.copy(img_yolo)
    img_combine_rect = np.copy(img)
    img_light_rect = np.copy(img)
    
    img_h, img_w, img_c = img.shape
    light_mask = np.zeros((img_h,img_w,img_c))
    light_mask.fill(255.0)
    
    ##　feature fusion ##

    # Light detection
    for box in zip(g.get_group(filename)['position'], g.get_group(filename)['predict']):                
        if box[1] != 0:
            box = json.loads(box[0]) 
            x, y, w, h = box
            light_mask[y:y+h,x:x+w] = [0,0,0]
            img_light_rect = cv2.rectangle(img_light_rect, (x, y), (x+w, y+h), [0,0,255], 3,cv2.LINE_AA)
    data=[]
    with open(yolo_path+'out/tlchia-dataset-v2_day/yolo_data/'+str(filename)+'.json') as f:
        data = json.load(f)
        data = data.replace("'", '"')
        data = json.loads(data)


    # Vehicle Detection
    
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

        # makeing mask
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
                    
        save_path = 'img/demo-out/3/img_result_light_mask/'
        print(save_path+str(filename)+".png")
        cv2.imwrite(save_path+'combine/'+str(filename)+".png", img_result_combine)
        cv2.imwrite(save_path+'yolo/'+str(filename)+".png", img_result_yolo)
        cv2.imwrite(save_path+'light/'+str(filename)+".png", img_result_light)
        cv2.imwrite(save_path+'combine_rect/'+str(filename)+".png", img_combine_rect)
        cv2.imwrite(save_path+'yolo_rect/'+str(filename)+".png", img_yolo_rect)
        cv2.imwrite(save_path+'light_rect/'+str(filename)+".png", img_light_rect)
                        