


import cv2
import os,sys
import glob
import shutil

parentdir = '../'
sys.path.insert(0,parentdir) 

from utils.files import getBaseName, createFolder

try:
    shutil.rmtree('/home/alanhc-school/Downloads/research/research-beta/architecture0714/img/demo-out') 
except:
    print('folder clean.')


files = glob.glob('../../../dataset/highway_video/*')
video_path = files[1]
print(files)

filename = getBaseName(video_path)[0]
createFolder('../img')
createFolder('../img/demo-out')
createFolder('../img/demo-out/'+filename)
createFolder('../img/demo-out/'+filename+'/origin')


save_path = '../img/demo-out/'+filename+'/'
cyclegan_path = '/home/alanhc-school/Desktop/CycleGAN-Tensorflow-2/'

vidcap = cv2.VideoCapture(video_path)
success,image = vidcap.read()
count = 0
while success:
    try:
        # save frame as JPEG file      
        success,image = vidcap.read()
        cv2.imshow('frame',image)
        img_256 = cv2.resize(image, (256,256), interpolation = cv2.INTER_AREA)
        
        cv2.imwrite(save_path+'origin/'+str(count)+'.jpg', image)
        #cv2.imwrite(save_path+'origin_256/'+str(count)+'.jpg', img_256)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print('Read %d new frame: ', count, success)
        count += 1
    except:
        break
    
# copy to cyclegan as input
try:
    shutil.rmtree(cyclegan_path+'datasets/tlchia-dataset-v2/testA/') 
except:
    print('folder clean.')


shutil.copytree('../img/demo-out/'+filename+'/origin', cyclegan_path+'datasets/tlchia-dataset-v2/testA')
