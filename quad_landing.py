import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from pyramid_image import *
from template_matcher import *

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

def build_ovrs(img, n_ovrs, scale):
    cur_img = img
    cur_scale = 1
    layers_dict = {}
    for k in range(n_ovrs):
        cur_scale = cur_scale * scale
        img_dwnsmpl = cv2.resize(cur_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        layers_dict[cur_scale] = img_dwnsmpl
        cur_img = img_dwnsmpl
    return layers_dict

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="Path to the video.")
ap.add_argument("-t", "--target", required=True, help="Path to target image.")
ap.add_argument("-c", "--split", required=False, help="Save each frame.")
args = vars(ap.parse_args())

# Open video file.
cap = cv2.VideoCapture(args["video"])
# Open target image.
target_img = cv2.imread(args["target"])

# Create vector of pyramids
# pyr_iamge = PyramidImage(target_img)
# pyr_iamge.build_ovr(4, 0.5)
layers_dict = build_ovrs(target_img, n_ovrs=4, scale=0.5)

# Check if camera opened successfully.
if (cap.isOpened() == False):
  print("Error opening video stream or file")

# OPtical flow parameters.
# Params for ShiTomasi corner detection.
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
# Parameters for Lucas Kanade optical flow.
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take the first frame and find corners in it.
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

i = 0
ret = True
while(cap.isOpened() and ret is True):
  # Capture frame-by-frame
  ret, frame = cap.read()
  frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)

  # The first frame.
  if i == 0:
    prev_frame = frame

  if ret == True:
    if i % 3 == 0:
      # Find the best Harris features.
      p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
      i = 0
      # Lukas-Kanade method
      p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
      # cv2.imshow('Flow', flow)

      # Select good points.
      good_new = p1[st==1]
      good_old = p0[st==1]

      # Draw the tracks
      mask = np.zeros_like(old_frame)
      for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        print(np.sqrt((a-c) * (a-c) + (b-d) * (b-d)))
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        tmp_frame = frame.copy()
        img = cv2.add(tmp_frame, mask)

      cv2.imshow('frame', img)

    i = i + 1
    prev_frame = frame
    old_gray = frame_gray
    find_template_SIFT(prev_frame, layers_dict[0.5])#target_img)

    # Press Q on keyboard to exit.
    if cv2.waitKey(25) & 0xFF == ord('q') or ret is False:
      break

  # Break the loop.
  else:
    break

# When everything done, release the video capture object.
cap.release()

# Closes all the frames.
cv2.destroyAllWindows()
