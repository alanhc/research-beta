import pandas as  pd
import cv2
import numpy as np
from utils.files import getBaseName, createFolder
import json


def rect_cross(boxes):
    if boxes[0][2]>boxes[1][0] or boxes[0][3]<boxes[1][1]:
        return True
    else:
        return False

def mid_pos(boxes):
    pos =[]
    for i in range(4):
        pos.append( (boxes[0][i]+boxes[1][i])//2 )
    pos[2] = pos[2] - pos[0]
    pos[3] = pos[3] - pos[1]
    return pos

data = pd.read_csv('./demo-result-before.csv')
g = data.groupby(["filename"])


# smooth the data
roi_names = g.count().index
for i in range(len(roi_names)-1):
    #print("==", data['filename']==int(roi_names[i]) )
    #print("===", data)
    frame_data_old = data[ data['filename']==int(roi_names[i]) ]
    frame_data_now = data[ data['filename']==roi_names[i+1] ]
    for old_data in frame_data_old.head().iterrows():
        for now_data in frame_data_now.head().iterrows():
            filename = now_data[1]['filename']
            o_data = json.loads(str(old_data[1]['position']))
            n_data = json.loads(str(now_data[1]['position']))
            r1_x1, r1_y1, r1_w, r1_h = o_data
            r1_x2 = r1_x1 + r1_w
            r1_y2 = r1_y1 + r1_h
            r2_x1, r2_y1, r2_w, r2_h = n_data
            r2_x2 = r2_x1 + r2_w
            r2_y2 = r2_y1 + r2_h
            boxes = []
            boxes.append([r1_x1,r1_y1, r1_x2, r1_y2])
            boxes.append([r2_x1,r2_y1, r2_x2, r2_y2])
            boxes.sort()
            is_cross = rect_cross(boxes)
            if is_cross:
                x,y,w,h = mid_pos(boxes)
                name_check = data['filename']==filename
                pos_check = data['position'] == str(n_data)
                data.loc[pos_check,'position'] =  str([x,y,w,h])
save_data =  pd.DataFrame(data, columns=['filename', 'position'])
save_data.to_csv('demo-result-after.csv')  

g = save_data.groupby(["filename"])

print(data.shape)

createFolder('img/demo-out/3/img_result_light_mask')
names = ['yolo', 'yolo_rect', 'light', 'light_rect', 'combine', 'combine_rect']
for i in range(len(names)):
    createFolder('img/demo-out/3/img_result_light_mask/'+names[i])

yolo_path = '/home/alanhc-school/yolo/darkflow/'
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
    #print(img.shape)
    combine_mask = np.zeros((img_h,img_w,img_c))
    combine_mask.fill(255.0)
    combine_mask = combine_mask.astype('uint8')
    light_mask = np.zeros((img_h,img_w,img_c))
    light_mask.fill(255.0)
    

    # Light detection
    for box in zip(g.get_group(filename)['position']):                
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
                        