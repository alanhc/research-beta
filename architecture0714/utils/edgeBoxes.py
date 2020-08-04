import cv2 
import numpy as np
from utils.color_filter import  binary_color_filter

model = 'model.yml'
windows = 30
def Edgeboxes(img_gray, img_origin, color, img_roi_combine, state, filename, base):
    
    img_origin = np.copy(img_origin)
    img_gray = np.copy(img_gray)
    

    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model)

    img_gray = img_gray.astype('uint8')

    

    rgb_im = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)
    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)
    
    edge_boxes = cv2.ximgproc.createEdgeBoxes(minBoxArea=windows, maxBoxes=100)
   
    
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)

    if len(boxes) > 0:
        boxes_scores = zip(boxes)
        for b_s in boxes_scores:
            box = b_s[0]
            x, y, w, h = box
            
            img_origin = cv2.rectangle(img_origin, (x, y), (x+w, y+h), color, 1,cv2.LINE_AA)
            img_roi_combine = cv2.rectangle(img_roi_combine, (x, y), (x+w, y+h), color, 1,cv2.LINE_AA)

            
            
            
            

    return img_origin, img_roi_combine, boxes