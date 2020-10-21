import cv2
import numpy as np
import os
from convention import diffAndMask, alignImages


def registrate(imflawpath, imrefpath):
    im = cv2.imread(imflawpath)
    imReference = cv2.imread(imrefpath)

    imgReg = alignImages(im, imReference)
    cv2.imwrite(os.path.join(os.environ.get('UPLOAD_DIR'), 'reg.jpg'), imgReg)

def detect(imregpath, imrefpath):
    imReg = cv2.imread(imregpath)
    imReference = cv2.imread(imrefpath)
    
    imDet, imCrop = diffAndMask(imReg, imReference)
    cv2.imwrite(os.path.join(os.environ.get('UPLOAD_DIR'), 'det.jpg'), imDet)
    cv2.imwrite(os.path.join(os.environ.get('UPLOAD_DIR'), 'crop.jpg'), imCrop)