import cv2
import numpy as np


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
           

def combinate_image(im1:np.ndarray, im2:np.ndarray):
    im = im1.copy()
    im[:,:, 2] = im2[:,:,2]
    return im

if __name__ == "__main__":
    im1 = cv2.imread('uploads/base.jpg')
    im2 = cv2.imread('uploads/reg.jpg')
    im = combinate_image(im1, im2)
    cv2.imwrite('test.jpg', im)