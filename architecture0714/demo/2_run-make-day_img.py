import os
import shutil

try:
    shutil.rmtree('/home/alanhc-school/Desktop/CycleGAN-Tensorflow-2/output/tlchia-dataset-v2/samples_testing')
except:
    print('folder not exist.')

cyclegan_path = '/home/alanhc-school/Desktop/CycleGAN-Tensorflow-2/'
os.chdir(cyclegan_path)
os.system('CUDA_VISIBLE_DEVICES=0,1 python test.py --experiment_dir ./output/tlchia-dataset-v2')


import os
import glob
import cv2
import shutil

yolo_path = '/home/alanhc-school/yolo/darkflow/'
cyclegan_path = '/home/alanhc-school/Desktop/CycleGAN-Tensorflow-2/'
os.chdir(cyclegan_path)
dataset_name = 'tlchia-dataset-v2'
files = glob.glob('output/'+dataset_name+'/samples_testing/A2B/*.jpg')
print(files)
save_path_origin = 'output/'+dataset_name+'/samples_testing/origin/'
save_path_A = 'output/'+dataset_name+'/samples_testing/A/'
save_path_B = 'output/'+dataset_name+'/samples_testing/B/'

if not os.path.exists(save_path_origin):
    os.mkdir(save_path_origin)
if not os.path.exists(save_path_A):
    os.mkdir(save_path_A)
if not os.path.exists(save_path_B):
    os.mkdir(save_path_B)


i=0
for f in files:   
    path = f
    base = os.path.basename(path)
    filename = os.path.splitext(base)[0]
    print(filename)

    img = cv2.imread(f,1)
    h,w,c = img.shape
    img_origin = img[:,:256]
    img_A = img[:,256:256*2]
    img_B = img[:,256*2:256*3]
    

    

    cv2.imshow('frame',img)
    
    
    cv2.imwrite(save_path_A+filename+'.jpg',img_A)
    cv2.imwrite(save_path_B+filename+'.jpg',img_B)
    
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break
    i+=1
try:
    shutil.rmtree(yolo_path+dataset_name+'_day')
except:
    print('folder not exist.')
shutil.copytree(save_path_A, yolo_path+dataset_name+'_day')