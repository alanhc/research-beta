import glob
import re
import cv2
import sys
import numpy as np
import time
parentdir = '../'
sys.path.insert(0,parentdir) 

from utils.files import getBaseName, createFolder

video_path = '../img/demo-out/3/'
"""
folder_names = [
 'img_basic_H', 'img_basic_hsv', 'img_result_white_edgeboxes', 'img_result_roi_combine', 'img_color_white_binary', 'img_basic_S',
 'img_light_detection_nakagami_norm_th', 'img_color_white_filted', 'img_basic_V', 'img_light_detection_gray_th', 'img_color_white_contour',
 'img_light_detection_nakagami_norm', 'img_basic_gray', 'img_color_white_mor', 'img_light_detection_nakagami_norm_th_clip', 
 'origin', 'img_result_light_mask/combine', 'img_result_light_mask/combine_rect', 'img_result_light_mask/light', 'img_result_light_mask/light_rect', 
 'yolo', 'yolo_rect']
"""
folder_names = ['img_basic_H']




for folder in folder_names:
    image_path = '../img/demo-out/3/'+folder+'/'
    files = glob.glob(image_path+'*.png')

    i=0
    filenames = []
    for f in files:
        filename = getBaseName(f)[0]
        filenames.append(int(filename))
        i+=1
    print(folder, " has %d files"%i)

    # batch process
    batch = i//3000 + 1
    batch_filenames = []
    files = []
    ct=0
    for f in sorted(filenames):
        if ct >3000:
            batch_filenames.append(files)
            ct=0
            files=[]
        files.append(f)
        ct+=1
    print(len(batch_filenames))
    for j in range(len(batch_filenames)):
        print(len(batch_filenames[j]))
    

    #print(len(batch_filenames))
    """
    img = cv2.imread(image_path+'1.png')
    h,w,c = img.shape
    #print(h,w)
    size = (w,h)
    """
    
    #frames = []
    bI=0
    for f in sorted(batch_filenames[bI]):
        frames = np.zeros((len(batch_filenames[bI]),h,w,c), dtype='uint8')
        print(frames.shape)
        bI+=1
        """
        
        tStart = time.time()
        img = cv2.imread(image_path+str(f)+'.png')
        frames[ct] = img
        #frames.append(img)
        ct+=1
        tEnd = time.time()

        pass_t = tEnd - tStart
        remain_t = (i-ct) * pass_t
        if ct%8==0:
            print("Making frames... ["+str(ct)+"/"+str(i)+"]"+" speed:{:.2f}/sec ".format(pass_t)+" remain:{:.2f}".format(remain_t)+"\r", end="")
        
        save_name_list = folder.split('/')
        save_name=""
        for s in save_name_list:
            save_name+=s
        out = cv2.VideoWriter(video_path+save_name+"batch-"+str(j)+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
        for f in range(i):
            
            out.write(frames[f])
        out.release()
        bI+=1
        """
print('done.')