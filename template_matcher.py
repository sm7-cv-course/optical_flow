import numpy as np
import matplotlib.pyplot as plt
import cv2


def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, list(kp_pairs)


def find_template_SIFT(img, tplt, show_res=true):
    """
    Tempalte finder based on SIFT method.
    """
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    if len(img.shape) == 3:
        tplt_gray = cv2.cvtColor(tplt, cv2.COLOR_BGR2GRAY)
    else:
        tplt_gray = tplt
    # img_gray = np.float32(img_gray)
    # tplt_gray = np.float32(tplt_gray)

    # Initialize SIFT detector.
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img_gray, None)
    kp2, des2 = sift.detectAndCompute(tplt_gray, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = np.zeros(img_gray.shape)
    img3 = cv2.drawMatchesKnn(img, kp1, tplt, kp2, good, img3, flags=2)

    p1, p2, kp_pairs = filter_matches(kp1, kp2, matches)

    # plt.imshow(img3), plt.show()
    if show_res:
        cv2.imshow('SIFT', img3)
    # if len(p1) >= 4:
    M, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    return M


def find_template_bf(img, tplt):
    """
    Tempalte finder based on brutforce method.
    """
    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)


def find_template_Harris(img, tplt):
    """
    Tempalte finder based on Harris detector.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tplt_gray = cv2.cvtColor(tplt, cv2.COLOR_BGR2GRAY)

    img_gray = np.float32(img_gray)
    tplt_gray = np.float32(tplt_gray)
    img_dst = cv2.cornerHarris(img_gray, 2, 3, 0.04)
    tplt_dst = cv2.cornerHarris(tplt_gray, 2, 3, 0.04)
