import cv2
import numpy as np
from utils.color_filter import binary_color_filter

def make_feature(boxes, img_ground, img_ground_mask, state, img_H,img_yolo_b):
    answer_color = [
                    [255,0,0],[0,255,0],
                    [0,255,255],[255,0,255], [0,0,255], [0,0,0]
    ]
    features=[]
    answers=[]
    if len(boxes) > 0:
        boxes_scores = zip(boxes)
        for b_s in boxes_scores:
            box = b_s[0]
            x, y, w, h = box
            img_ground_filted = cv2.bitwise_or(img_ground, img_ground, mask=img_ground_mask)
            tmp = img_ground_filted[y:y+h,x:x+w]
            t ,ct= np.unique(tmp.reshape(-1, tmp.shape[2]), axis=0, return_counts=True)
            
            ### make answer
            
            yolo_and_ground = np.bitwise_and(img_ground_mask[y:y+h,x:x+w] ,img_yolo_b[y:y+h,x:x+w])
            area_yolo_and_ground = (yolo_and_ground//255).sum()
            """
            cv2.imshow("yolo_and_ground", yolo_and_ground)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """

            ROI_combine= area_yolo_and_ground/(w*h)
            center_line = img_H[y+h//2, x:x+w]
            ROI_center_std = np.std(center_line)
            ROI_center_min = np.amin(center_line)
            ROI_center_area = w*h
            ROI_height = y
            feature = [ROI_combine,ROI_center_min,ROI_center_std,ROI_height,ROI_center_area]
            
            ### make feature
            if state=='train':
                answer=""
                l_t = t.tolist()
                l_c = ct.tolist()
                if [0,0,0] in l_t:
                    if ct.shape[0]!=1:
                        black_idx = l_t.index([0,0,0])
                        
                        l_t.pop(black_idx)
                        l_c.pop(black_idx)

                        max_idx = l_c.index(max(l_c))
                        max_color = l_t[max_idx]
                        try:
                            answer = answer_color.index(max_color)
                        except:
                            print('label error')
                            answer = 5
                        
                        t = np.array(t)
                        ct = np.array(ct)
                    else:
                        try:
                            answer = answer_color.index(t[0].tolist())
                        except:
                            print('label error')
                            answer = 5
                else:
                    max_idx = l_c.index(max(l_c))
                    max_color = l_t[max_idx]
                    answer = answer_color.index(max_color)

            
            features.append(feature)
            answers.append(answer)
            #print(feature,answer)
            """
            cv2.imshow("tmp", tmp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """
            
    return features, answers

"""
    cv2.imshow("img_red_contour_cliped", img_red_contour_cliped)
    #cv2.imshow("img_white_filted_gray", img_white_filted_gray)
   
    
"""