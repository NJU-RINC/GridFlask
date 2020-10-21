import cv2
import numpy as np

MAX_FEATURES = 5000
GOOD_MATCH_PERCENT = 0.45


def alignImages(im1, im2):
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(MAX_FEATURES)

    kp1 = orb.detect(im1Gray)
    kp1, des1 = orb.compute(im1Gray, kp1)
    kp2 = orb.detect(im2Gray)
    kp2, des2 = orb.compute(im2Gray, kp2)

    matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)
    matches = matcher.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[numGoodMatches:-1]
    # img3 = cv2.drawMatches(im1Gray,kp1,im2Gray,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plotImages(img3)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 0.5)

    h, w = im2Gray.shape
    im1Reg = cv2.warpPerspective(im1, M, (w, h))
    # plotImages(im2, im1Reg, im1, titles=['base', 'reg', 'flaw'])

    return im1Reg


def diffAndMask(imReg, imReference):
    preframe = cv2.cvtColor(imReference, cv2.COLOR_BGR2GRAY)

    curframe = cv2.cvtColor(imReg, cv2.COLOR_BGR2GRAY)
    curframe = cv2.absdiff(curframe, preframe)
    ret, curframe = cv2.threshold(curframe, 120, 255.0, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    curframe = cv2.erode(curframe, kernel)
    curframe = cv2.dilate(curframe, kernel)

    contours, hierarchy = cv2.findContours(curframe, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)

    x, y, w, h = cv2.boundingRect(
        max(contours, key=lambda x: cv2.contourArea(x)))
    imCrop = imReg.copy()
    imCrop = imCrop[y:y+h, x:x+w]
    cv2.rectangle(imReg, (x, y), (x + w, y + h), [0, 0, 255], 3)

    return imReg, imCrop