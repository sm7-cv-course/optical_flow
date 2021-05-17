import argparse
import cv2 #as cv
import numpy as np
from template_matcher import *
#https://github.com/abhiTronix/vidgear
from vidgear.gears import VideoGear
from vidgear.gears import WriteGear

scale = 0.5

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="Path to the video.")
ap.add_argument("-c", "--split", required=False, help="Save each frame.")
args = vars(ap.parse_args())

stream_stab = VideoGear(source=args["video"], stabilize=True).start()

frame = stream_stab.read()
frame_gray1 = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
frame_gray1 = cv2.resize(frame_gray1, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
frame_gray1 = cv2.GaussianBlur(frame_gray1, (5, 5), 0)
frame_prev = frame_gray1

# loop over
while True:

    # read stabilized frames
    frame_stab = stream_stab.read()
    frame_gray = cv2.cvtColor(frame_stab, cv2.COLOR_RGBA2GRAY)

    frame_stab_small = cv2.resize(frame_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    frame_stab_blur = cv2.GaussianBlur(frame_stab_small, (5, 5), 0)

    M = find_template_SIFT(frame_stab_blur, frame_gray1)

    height, width = frame_stab_blur.shape
    frame_tr = cv2.warpPerspective(frame_stab_blur, M, (width, height))

    # Check for stabilized frame if Nonetype
    if frame_stab is None:
        break

    type = cv2.THRESH_BINARY
    diff = np.fabs(np.float32(np.float32(frame_gray1) - np.float32(frame_tr)))
    ret, img_bw = cv2.threshold(diff, 15, 255, type)

    # Show output window
    cv2.imshow("Stabilized BW", img_bw)#(frame_gray1 - frame_tr))

    frame_prev = frame_stab_blur

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Close output window
cv2.destroyAllWindows()
stream_stab.stop()
