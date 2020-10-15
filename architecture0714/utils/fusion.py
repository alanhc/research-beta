import cv2
import numpy as np
from utils.color_filter import binary_color_filter


def make_feature(boxes, img_ground, img_ground_mask, state, img_S,img_yolo_b, filename, version=None, img=None, boxI=None ):
    
    answer_color = [
                    [255,0,0],[0,255,0],   #0 1 
                    [0,255,255],[255,0,255], # 2 3
                    [0,0,255], [0,0,0], [255,255,0]  #456
    ]
    features=[]
    answers=[]
    if len(boxes) > 0:
        boxes_scores = zip(boxes)
        
        for b_s in boxes_scores:
            box = b_s[0]
            x, y, w, h = box
            
            if state=='train':
                img_ground_filted = cv2.bitwise_or(img_ground, img_ground, mask=img_ground_mask)
                tmp = img_ground_filted[y:y+h,x:x+w]
                t ,ct= np.unique(tmp.reshape(-1, tmp.shape[2]), axis=0, return_counts=True)
            
            
            yolo_and_edgebox = img_yolo_b[y:y+h,x:x+w]
            #cv2.imshow("img_yolo_b", yolo_and_edgebox)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            ### make feature
            area_yolo_and_edgebox = (yolo_and_edgebox//255).sum()
            
            ROI_combine= area_yolo_and_edgebox/(w*h)
            center_line = img_S[y+h//2, x:x+w]
            ROI_center_std = np.std(center_line)
            ROI_center_min = np.amin(center_line)
            ROI_center_area = w*h
            ROI_height = y
            feature = [filename,ROI_combine,ROI_center_min,ROI_center_std,ROI_height,ROI_center_area, [x,y,w,h]]
            features.append(feature)
            if ROI_combine==0:
                continue
            
            ### make answer
            if state=='train':
                yolo_and_ground = np.bitwise_and(img_ground_mask[y:y+h,x:x+w] ,img_yolo_b[y:y+h,x:x+w])
                area_yolo_and_ground = (yolo_and_ground//255).sum()
                
            
            

            
            
            ### make answer
            if state=='train':
                answer=""
                l_t = t.tolist()
                l_ct = ct.tolist()
                #print(l_t)
                #print(l_ct)
                
                black_area=0
                # pop black
                if [0,0,0] in l_t:
                    black_idx = l_t.index([0,0,0])
                    black_area = l_ct[black_idx]
                    l_t.pop(black_idx)
                    l_ct.pop(black_idx)
                if [0,0,255] in l_t:
                    red_idx = l_t.index([0,0,255])
                    l_t.pop(red_idx)
                    l_ct.pop(red_idx)
                
                if len(l_t)==0:
                    answer=-1    
                else:
                    max_idx = l_ct.index(max(l_ct))
                    max_color = l_t[max_idx]
                    answer = answer_color.index(max_color)
            
                    if answer in [0,1]:
                        answer = 0
                    elif answer in [2,3]:
                        answer = 1
                    else:
                        answer = -1
                
                if version=='v6':
                    if answer in [0,1,2,3]:
                        answer = 1 
                    else:
                        answer=0
                elif version=='v7-hand':
                    
                    img_show = np.copy(img)
                    #img_show[y:y+h,x:x+w] = [0,0,255]
                    img_show = cv2.rectangle(img_show, (x, y), (x+w, y+h), [0,0,255], 3,cv2.LINE_AA)
                    print("boxI:",boxI,'[',x,y,w,h,']')
                    
                    cv2.imshow("img_show-"+str(boxI), img_show)
                    key = cv2.waitKey(0)
                    if key==32:
                        answer=0
                    else:
                        answer=1
                    print(answer)
                    #if boxI>1335:
                    #    key = cv2.waitKey(0)
                    
                    cv2.destroyAllWindows()
                
                if boxI!=None:
                    boxI+=1
                    
                    
                    #print(max_color)
                #print('answer:',answer)

                

            """
            #print('label:',answer,black_area, l_ct[max_idx])
            print('iou:',ROI_combine,'answer:',answer)
            cv2.imshow("tmp", tmp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """
            
            if state=='train':
                features.append(feature)
                answers.append(answer)
                print(feature,answer)
            """
            cv2.imshow("tmp", tmp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """
    
    if state=='test':
        if boxI!=None:
            return features, boxI
        else:
            return features
    
    if boxI!=None:
        return features, answers, boxI
    else:
        return features, answers

"""
    cv2.imshow("img_red_contour_cliped", img_red_contour_cliped)
    #cv2.imshow("img_white_filted_gray", img_white_filted_gray)
   
    
"""