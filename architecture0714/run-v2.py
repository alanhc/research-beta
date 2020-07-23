
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np

########## Method import ##########
from utils.files import getBaseName, createFolder
from utils.color_filter import Euclidean_filter
from utils.enhancement import Nakagami_image_enhancement, contour_Detection, clip_center
from utils.edgeBoxes import Edgeboxes


base = '../svm/dataset_100/dataset_100/'

save = True

def main(frame_path):
    print(frame_path)
    filename, f_type = getBaseName(frame_path)
    createFolder('img')
    createFolder('img/out')
    createFolder('img/out/'+filename)
    save_path = 'img/out/'+filename+'/'

    img = cv2.imread(frame_path, 1)
    h, w, c = img.shape
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_H = img_hsv[:,:,0]
    img_S = img_hsv[:,:,1]
    img_V = img_hsv[:,:,2]
    
    [b,g,r] = cv2.split(img)
    
    img_red_filted = Euclidean_filter(img, 1,150,0,[255,0,0], [b,g,r], save_path)  #img, percent,up_threshold,lower_threshold color(RGB), save_path
    img_white_filted = Euclidean_filter(img, 1,255,7, [255,255,255], [b,g,r], save_path)  #img, percent, color(RGB), save_path
    
    
    img_red_filted_gray = cv2.cvtColor(img_red_filted, cv2.COLOR_BGR2GRAY)
    img_white_filted_gray = cv2.cvtColor(img_white_filted, cv2.COLOR_BGR2GRAY)
    
    img_red_filted_gray_max = img_red_filted_gray.max()
    img_white_filted_gray_max = img_white_filted_gray.max()
    

    img_red_filted_gray_norm = img_red_filted_gray/img_red_filted_gray_max
    img_white_filted_gray_norm = img_white_filted_gray/img_white_filted_gray_max

    img_red_nakagami = Nakagami_image_enhancement(img_red_filted_gray_norm, 3) #gray_img, kernel_size
    img_white_nakagami = Nakagami_image_enhancement(img_white_filted_gray_norm, 3) #gray_img, kernel_size
    
    
    img_red_nakagami_cliped = clip_center(img_red_nakagami, int(h/3), int(h)), # img, y_up, y_down
    img_white_nakagami_cliped = clip_center(img_white_nakagami, int(h/3), int(h)), # img, y_up, y_down
    img_red_nakagami_cliped = img_red_nakagami_cliped[0]
    img_white_nakagami_cliped = img_white_nakagami_cliped[0]
   
    
    

    img_red_contour = contour_Detection(img_red_nakagami_cliped)
    img_white_contour = contour_Detection(img_white_nakagami_cliped)

    
    
    
    #print(type(img_red_contour_cliped[0]))
    
    """
    cv2.imshow("img_red_contour_cliped", img_red_contour_cliped)
    #cv2.imshow("img_white_filted_gray", img_white_filted_gray)
    #cv2.imshow("img_red_nakagami", img_red_nakagami)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    img_roi_combine = np.copy(img)
    img_red_edgeboxes,img_roi_combine = Edgeboxes(img_red_contour, img, [0,255,0], img_roi_combine) #contour, img_origin, box color(BGR)
    img_white_edgeboxes,img_roi_combine = Edgeboxes(img_white_contour, img, [255,0,0], img_red_edgeboxes) #contour, img_origin, box color(BGR)
    
   

    

    if save:
        cv2.imwrite('img/out/'+filename+'/'+filename+'_origin.png', img)
        cv2.imwrite('img/out/'+filename+'/'+filename+'_img_H.png', img_H)
        cv2.imwrite('img/out/'+filename+'/'+filename+'_img_S.png', img_S)
        cv2.imwrite('img/out/'+filename+'/'+filename+'_img_V.png', img_V)
        cv2.imwrite('img/out/'+filename+'/'+filename+'_img_red_filted.png', img_red_filted)
        cv2.imwrite('img/out/'+filename+'/'+filename+'_img_white_filted.png', img_white_filted)

        cv2.imwrite('img/out/'+filename+'/'+filename+'_img_red_filted_gray.png', img_red_filted_gray)
        cv2.imwrite('img/out/'+filename+'/'+filename+'_img_red_nakagami.png', img_red_nakagami)
        cv2.imwrite('img/out/'+filename+'/'+filename+'_img_white_nakagami.png', img_white_nakagami)
        
        cv2.imwrite('img/out/'+filename+'/'+filename+'_img_red_nakagami_cliped.png', img_red_nakagami_cliped)
        cv2.imwrite('img/out/'+filename+'/'+filename+'_img_white_nakagami_cliped.png', img_white_nakagami_cliped)
        
        cv2.imwrite('img/out/'+filename+'/'+filename+'_img_red_contour.png', img_red_contour)
        cv2.imwrite('img/out/'+filename+'/'+filename+'_img_white_contour.png', img_white_contour)

        
        

        cv2.imwrite('img/out/'+filename+'/'+filename+'_img_red_edgeboxes.png', img_red_edgeboxes)
        cv2.imwrite('img/out/'+filename+'/'+filename+'_img_white_edgeboxes.png', img_white_edgeboxes)
        cv2.imwrite('img/out/'+filename+'/'+filename+'_img_roi_combine.png', img_roi_combine)



        #cv2.imwrite('img/out/'+filename+'/'+filename+'_img_red.png', r)
        



    

if __name__ == '__main__':
    files = glob.glob(base+"origin/*")
    for f in sorted(files):
        main(f)
   