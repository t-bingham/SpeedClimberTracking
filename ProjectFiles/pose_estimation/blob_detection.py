# blob_detection.py

import cv2
import numpy as np

cap = cv2.VideoCapture('climbing3.mp4')  # Open the first camera connected to the computer.

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Set thresholds for the image binarization.
params.minThreshold = 10
params.maxThreshold = 200
params.thresholdStep = 10

# Filter by colour.
params.filterByColor = False
params.blobColor = 200

# Filter by Area.
params.filterByArea = True
params.minArea = 5000
params.maxArea = 50000

# Filter by Circularitymask
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexityx-s
params.filterByConvexity = True
params.minConvexity = 0.5

# Filter by Inertiamask
params.filterByInertia = True
params.minInertiaRatio = 0.75

detector = cv2.SimpleBlobDetector_create(params)

while True:
    ret, frame = cap.read()  # Read an image from the frame.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints = detector.detect(gray)

    for point in keypoints:
        x = point.pt[0]
        y = point.pt[1] #i is the index of the blob you want to get the position
        s = 100
        #print(x, y, s)
    rec_mask = np.zeros(frame.shape[:2], np.uint8)
    #cv2.rectangle(rec_mask,(int(x-s),int(y-s)),(int(x+s),int(y+s)),(0,255,0),3)
    rec_mask[int(y-s):int(y+s),int(x-s):int(x+s)] = 255
    mask = cv2.bitwise_and(frame, frame, mask=rec_mask)

    detected = cv2.drawKeypoints(frame, keypoints, None, (0,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('detected', mask)  # Show the image on the display.
    if cv2.waitKey(100) & 0xFF == ord('q'):  # Close the script when q is pressed.
        break

# Release the camera device and close the GUI.
cap.release()
cv2.destroyAllWindows()
