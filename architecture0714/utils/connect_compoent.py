import numpy as np
import cv2

def ccl(img):
    num_labels, labels_im = cv2.connectedComponents(img)
    img_ccl_origin = np.copy(labels_im)
    label_hue = np.uint8(179*labels_im/np.max(labels_im))
    blank_ch = 255*np.ones_like(label_hue)
    label_hue = np.uint8(179*labels_im/np.max(labels_im))
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        # set bg label to black
    labeled_img[label_hue==0] = 0

    return img_ccl_origin,labeled_img,num_labels

def find_BoundingBox(img, img_origin, max_boxes=100, min_area=100):
    img_origin = np.copy(img_origin)
    h,w = img.shape
    im_max = img.max()
    
    boxes = []
    ct=0
    for index in range(1,im_max+1):
        if ct>max_boxes:
            break
        i,j = np.where(img==index)
        i = np.sort(i)
        j = np.sort(j)
        h_min = i[0]
        h_max = i[-1]
        w_min = j[0]
        w_max = j[-1]
        if (h_max-h_min)*(w_max-w_min)<min_area:
            continue
        cv2.rectangle(img_origin, (w_min,h_min), (w_max, h_max), (0,0,255), 2)
        
        boxes.append([w_min,h_min, w_max, h_max])
        #print(index,boxes[i])
        """contour method
        img_i_b = np.where(img==i, 255,0)
        contours, hierarchy = cv2.findContours(  img_i_b,
                                                        cv2.RETR_LIST,
                                                        cv2.CHAIN_APPROX_SIMPLE
                                                     )
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(img_origin, (x,y), (x+w, y+h), (0,0,255), 2)
        """
        ct+=1
    """
    cv2.imshow("img_origin", img_origin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    return img_origin, boxes