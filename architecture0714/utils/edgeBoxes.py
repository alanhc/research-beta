import cv2 
import numpy as np
# https://docs.opencv.org/master/d4/d0d/group__ximgproc__edgeboxes.html
model = 'model.yml'

def Edgeboxes(img_gray, img_origin, color, img_roi_combine):
    
    img_origin = np.copy(img_origin)
    img_gray = np.copy(img_gray)
    img_roi_combine = np.copy(img_roi_combine)

    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model)

    img_gray = img_gray.astype('uint8')

    

    rgb_im = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)
    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)
    
    edge_boxes = cv2.ximgproc.createEdgeBoxes(minBoxArea=10, maxBoxes=50)
    #edge_boxes.setMaxBoxes(windows)
    
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)

    if len(boxes) > 0:
        boxes_scores = zip(boxes)
        for b_s in boxes_scores:
            box = b_s[0]
            x, y, w, h = box
            
            img_origin = cv2.rectangle(img_origin, (x, y), (x+w, y+h), color, 1,cv2.LINE_AA)
            img_roi_combine = cv2.rectangle(img_roi_combine, (x, y), (x+w, y+h), color, 1,cv2.LINE_AA)



    return img_origin, img_roi_combine