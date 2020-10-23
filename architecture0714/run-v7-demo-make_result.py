import pandas as  pd
import cv2

data = pd.read_csv('./demo-result-after.csv')
g = data.groupby(["filename"])

print(data.shape)

img_path = '/home/alanhc-school/Downloads/research/research-beta/architecture0714/img/demo-out/3/img/'
for filename in g.count().index:

    img = cv2.imread(img_path+str(filename)+'.png', 1)
    img_h, img_w, img_c = img.shape
    #print(img.shape)
    combine_mask = np.zeros((img_h,img_w,img_c))
    combine_mask.fill(255.0)
    combine_mask = combine_mask.astype('uint8')
    
    
    #alpha = 0.5
    #beta = (1.0 - alpha)
    #img_result_combine = cv2.addWeighted(img, alpha, combine_mask, beta, 0.0)

