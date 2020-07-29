
import cv2
import glob
from utils.files import getBaseName, createFolder
base = '../../dataset/pic_100/origin/'


def main(frame_path):
    print(frame_path)
    filename, f_type = getBaseName(frame_path)
    createFolder(base+'jpg/')
    img = cv2.imread(frame_path, 1)
    img_small = cv2.resize(img, (256,256))
    cv2.imwrite(base+'jpg/'+filename+'.jpg', img_small)
    
    print(base+'jpg/'+filename+'.jpg')


if __name__ == '__main__':
    files = glob.glob(base+'*.bmp')
    for f in sorted(files):
        main(f)
   