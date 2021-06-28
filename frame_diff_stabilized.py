import argparse
import cv2 #as cv
import imutils
import numpy as np
from template_matcher import *

scale = 0.5

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="Path to the video.")
ap.add_argument("-c", "--split", required=False, help="Save each frame.")
ap.add_argument("-d", "--rotate", required=False, help="Rotate image.")
ap.add_argument("-s", "--source", required=False, help="Template image.")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["video"])
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

if args["source"] is not None:
    frame = cv2.imread(args["source"])
else:
    ret, frame = cap.read()

frame_gray1 = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
frame_gray1 = clahe.apply(frame_gray1)
frame_gray1 = cv2.resize(frame_gray1, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
frame_gray1 = cv2.GaussianBlur(frame_gray1, (5, 5), 0)

if args["rotate"] is not None:
    # Rotation angle in degree
    frame_gray1 = imutils.rotate(frame_gray1, angle=-float(args["rotate"]))
frame_prev = frame_gray1

# Loop over
while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
    frame_gray = clahe.apply(frame_gray)

    frame_small = cv2.resize(frame_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    frame_blur = cv2.GaussianBlur(frame_small, (5, 5), 0)

    if args["rotate"] is not None:
        # Rotation angle in degree
        frame_blur = imutils.rotate(frame_blur, angle = -float(args["rotate"]))

    M = find_template_SIFT(frame_blur, frame_gray1)

    height, width = frame_blur.shape
    frame_tr = cv2.warpPerspective(frame_blur, M, (width, height))

    # Check for frame if Nonetype
    if frame is None:
        break

    # Show output window
    thresh_type = cv2.THRESH_BINARY
    diff = np.fabs(np.float32(np.float32(frame_gray1) - np.float32(frame_tr)))
    ret, img_bw = cv2.threshold(diff, 50, 255, thresh_type)
    cv2.imshow("BW threshold", img_bw)

    frame_prev = frame_tr

    # Check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Close output windows
cv2.destroyAllWindows()
