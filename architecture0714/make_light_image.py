import pandas as pd
import glob
import cv2
import numpy as np
import json
from utils.files import getBaseName, createFolder
train_dataset = ['dataset_100', 'pic_100']
for dataset_name in train_dataset:
    base = '../../dataset/'+dataset_name+'/'
    dataset = "origin/"
    save_path = 'img/out/'+dataset_name+'/result_light_spot_image/'
    createFolder('img/out/'+dataset_name+'/result_light_spot_image')

    data = pd.read_csv(base+dataset+'data-6.csv')
    test_data = pd.read_csv(base+dataset+'data-6-test.csv')

    print(data.shape)
    print(test_data.shape)

    data = data[['position','fliename', 'answers']]
    test_data = test_data[['predict']]
    data = pd.concat([data,test_data], axis=1)
    print(data.head(3))

    g = data.groupby(["fliename"])


    data = pd.concat([data,test_data], axis=1)
    print(data.shape)
    print(data.head(3))
    print(g.count().index)


    for filename in g.count().index:
        img = cv2.imread(base+dataset+str(filename)+'.bmp', 1)
        h, w, c = img.shape
        mask = np.zeros((h,w,c))
        mask.fill(255.0)
        #print("group:",g.get_group(filename))
        for box in zip(g.get_group(filename)['position'], g.get_group(filename)['predict']):
            
            if box[1] != -1:
                box = json.loads(box[0]) 
                x, y, w, h = box
                mask[y:y+h,x:x+w] = [0,0,0]

        img = img.astype('uint8')
        mask = mask.astype('uint8')
        alpha = 0.8
        beta = (1.0 - alpha)
        img_result = cv2.addWeighted(img, alpha, mask, beta, 0.0)
        cv2.imwrite(save_path+str(filename)+".png", img_result)
    