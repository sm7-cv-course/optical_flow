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


#stream = VideoGear(args["video"], stabilize = True).start() # To open any valid video stream(for e.g device at 0 index)

stream_stab = VideoGear(source=args["video"], stabilize=True).start()
cap = cv2.VideoCapture(args["video"])

#frame = stream_stab.read()
ret, frame = cap.read()
frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
frame_prev = cv2.resize(frame_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

# loop over
while True:

    # read stabilized frames
    frame_stab = stream_stab.read()
    #ret, frame_stab = cap.read()
    frame_gray = cv2.cvtColor(frame_stab, cv2.COLOR_RGBA2GRAY)
    
    frame_stab_small = cv2.resize(frame_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    #dsize = (frame_stab.shape[1] / 2, frame_stab.shape[0] / 2))
    
    frame_stab_blur = cv2.GaussianBlur(frame_stab_small, (5, 5), 0)

    M = find_template_SIFT(frame_stab_blur, frame_prev)
    #M = cv2.getAffineTransform(pts1,pts2)

    #M = cv2.findHomography(pts1, pts2)
    #dst = cv2.perspectiveTransform(pts,M)
    height, width = frame_stab_blur.shape
    frame_tr = cv2.warpPerspective(frame_stab_blur, M, (width, height))

    # check for stabilized frame if Nonetype
    if frame_stab is None:
        break

    # Show output window
    #diff = np.zeros_like(frame_stab)
    #diff = 255 if (frame_prev - frame_stab) > 100 else 0
    diff = frame_prev - frame_tr#frame_stab_blur
    #img_gray = cv2.cvtColor(diff, cv2.COLOR_RGBA2GRAY)
    type = cv2.THRESH_BINARY
    #ret, img_bw = cv2.threshold(img_gray, 250, 255, type)
    cv2.imshow("Stabilized Frame", (frame_prev - frame_tr))
    
    frame_prev = frame_stab_blur
    
    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
        
# close output window
cv2.destroyAllWindows()

stream_stab.stop()
