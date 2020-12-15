import cv2
import numpy as np
import glob



w = 7
h = 7

objp = np.zeros((w*h,3), np.float32)

objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
print(objp)

world_3D_points = [] 
img_2D_points = []

images = glob.glob('*.jpg')

for fname in images:
    
    img = cv2.imread(fname)
    
    img_h, img_w, c = img.shape

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # 找到棋盤格角點
    ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
    # 如果找到足夠點對，將其存儲起來
    
    
    if ret == True:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        
        world_3D_points.append(objp)
        img_2D_points.append(corners)
        # 將角點在圖像上顯示
        corners= corners.reshape(49,2)
        
        cv2.drawChessboardCorners(img, (w,h), corners, ret)
        
        cv2.imwrite("out/out.png", img)
        #tmp=img
        
        ## S-T
        st_points = np.zeros((49,2))
        uv_points = np.zeros((49,2))
        
        i=0
        for c in corners:
            st_points[i][1] = -c[1] + img_h/2 #t
            st_points[i][0] = c[0] - img_w/2 #s
            
            # image resolution 72ppi 25.4
            uv_points[i][1] = st_points[i][1] * 25.4
            uv_points[i][0] = st_points[i][0] * 25.4
            
            
            print(uv_points[i], st_points[i], c)
            
        
            

