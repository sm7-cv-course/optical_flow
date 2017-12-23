import numpy as np
import cv2


def find_template(img, tplt):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tplt_gray = cv2.cvtColor(tplt, cv2.COLOR_BGR2GRAY)

    img_gray = np.float32(img_gray)
    tplt_gray = np.float32(tplt_gray)
    img_dst = cv2.cornerHarris(img_gray, 2, 3, 0.04)
    tplt_dst = cv2.cornerHarris(tplt_gray, 2, 3, 0.04)


sift = cv2.xfeatures2d.SIFT_create()
