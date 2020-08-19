import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import pandas as pd
import os
import json 
from sklearn.metrics import classification_report
import shutil


from utils.files import getBaseName, createFolder

base = '../../dataset/pic_100/'
dataset = "origin/"
save_path = base+'result_out'


if os.path.isdir(base+'result_out'):
    shutil.rmtree(base+'result_out')
createFolder(base+'result_out')


data = pd.read_csv(base+dataset+'data-2.csv')
test_data = pd.read_csv(base+dataset+'data-2-test.csv')

print('report:')
y_test = data['answers']
y_pred = test_data['predict']
print(classification_report(y_pred,y_test))

data = data[['fliename','position']]
test_data = test_data[['predict']]
merge = pd.concat([data, test_data], axis=1)


for i in range(merge.shape[0]):
    tStart = time.time()

    filename = merge['fliename'].iloc[i]
    position = merge['position'].iloc[i]
    label_pred = merge['predict'].iloc[i]
    
    filename = str(filename)
    if os.path.isfile(save_path+'/'+filename+'_predict_result.png'):
        
        img = cv2.imread(save_path+'/'+filename+'_predict_result.png', 1)
        
        
    else:
        img = cv2.imread(base+dataset+filename+'.bmp', 1)

    id_to_color = {
        0:[255,0,0],
        1:[0,0,255]
    }
    color=""
    
    if label_pred!=-1:
        res = json.loads(position) 
        
        x, y, w, h = res
       
        color = id_to_color[label_pred]
        #print(color, [x,y,w,h])
        img_fusion = cv2.rectangle(img, (x, y), (x+w, y+h), color,2,cv2.LINE_AA  )
        cv2.imwrite(save_path+'/'+filename+'_predict_result.png', img_fusion)
        
        
        tEnd = time.time()
        t_per_job = (tEnd - tStart)
        remain_job = merge.shape[0]-i
        remain_t = (remain_job*t_per_job)
        remain_t = int(remain_t)
        print('[', str(i)+'/'+str(merge.shape[0]),']',"It cost %f sec "% t_per_job, filename,str(remain_t),"\r",end='')
        #print('[', str(i)+'/'+str(merge.shape[0]),']',"It cost %f sec "% (tEnd - tStart), filename)
        
    



        